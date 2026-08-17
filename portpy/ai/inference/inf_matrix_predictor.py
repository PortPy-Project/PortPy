# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# ----------------------------------------------------------------------
"""
Beamlet-dose predictors for the predicted influence matrix.

``BeamletDosePredictor`` is the pluggable interface used by
``portpy.photon.inf_matrix_test.InfluenceMatrix`` in *predict* mode: given a
beam's geometry and its beamlets, it yields a CT-grid 3D dose volume per
beamlet. New models only need to implement this interface (or subclass
``RayUNetPredictor`` and override the input channels / normalization).

``RayUNetPredictor`` reproduces the validated research pipeline exactly:
  per beamlet -> ray-aligned patch (CT/d1/d1_out) -> 3D UNet -> unscale ->
  back-project to the CT grid.
"""
import abc
import numpy as np

from .ray_geometry import (
    get_rotation_matrix, get_ct_sitk_image, resample_to_grid,
    estimate_body_entry_distance_mm, build_ray_grid_with_beam_axes,
    compute_ray_patch_d1_d2_split, source_lps_from_beam, beamlet_center_lps,
)
from .gpu_resample import ct_to_ray_patch, sample_volume_at_world

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


class BeamletDosePredictor(abc.ABC):
    """Interface: produce a CT-grid 3D dose (Z,Y,X) for every beamlet of a beam."""

    @abc.abstractmethod
    def predict_beam(self, beam_geom: dict, beamlets: dict):
        """
        Yield ``(local_idx, dose_3d_ct_zyx)`` for each beamlet of one beam.

        beam_geom keys: gantry_angle, couch_angle, collimator_angle, SAD_mm,
                        iso_xyz_mm (3,), beam_id.
        beamlets keys : position_x_mm, position_y_mm, width_mm, height_mm (1D arrays).
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Default per-channel normalizers (match training in ANet_*_save_light)
# --------------------------------------------------------------------------
def _norm_ct(ct, lo=-1000.0, hi=3071.0):
    """CT HU clipped to [lo, hi] then scaled to [0, 1] (training Option A)."""
    return (np.clip(ct, lo, hi) - lo) / (hi - lo)


# Commercial TPS clinic CT calibration curve: HU -> Relative Electron Density.
# Must match RadiotherapyDataset._hu_to_relative_electron_density exactly.
_RED_HU_PTS = np.array([-3000.0, -1050.0, -1000.0, -900.0, -750.0, -500.0,
                        -250.0, 0.0, 100.0, 350.0, 600.0, 1500.0,
                        6424.0, 15000.0, 27780.0], dtype=np.float64)
_RED_VAL_PTS = np.array([0.000, 0.000, 0.000, 0.098, 0.249, 0.499,
                         0.750, 1.000, 1.059, 1.208, 1.356, 1.910,
                         3.790, 8.290, 15.000], dtype=np.float64)


def _norm_ct_density(ct, red_scale=3.79):
    """CT HU -> relative electron density / red_scale, clamped [0,1] (training Option B).

    Matches the 'ct_density' checkpoints (run name '..._ct_density'): the CT
    channel is fed as relative electron density rather than scaled HU.
    """
    ct = np.clip(ct, _RED_HU_PTS[0], _RED_HU_PTS[-1])
    red = np.interp(ct, _RED_HU_PTS, _RED_VAL_PTS)
    return np.clip(red / float(red_scale), 0.0, 1.0).astype(np.float32)


# Selectable CT channel normalizers (extend for new models).
CT_NORMALIZERS = {"hu": _norm_ct, "density": _norm_ct_density}


def _norm_div(arr, div):
    return np.clip(arr / div, 0.0, 1.0)


DEFAULT_CHANNELS = ("ct", "d1", "d1_out")
DEFAULT_NORM = {
    "ct": lambda a: _norm_ct(a),
    "d1": lambda a: _norm_div(a, 2000.0),
    "d1_out": lambda a: _norm_div(a, 2000.0),
    "d2": lambda a: _norm_div(a, 600.0),
    "body": lambda a: a.astype(np.float32),
    "proj_mask": lambda a: np.clip(a, 0.0, 1.0).astype(np.float32),
}


class RayUNetPredictor(BeamletDosePredictor):
    """
    Ray-aligned 3D UNet beamlet-dose predictor.

    Parameters
    ----------
    ct : PortPy CT object.
    body_mask_zyx : BODY mask on the CT grid (Z,Y,X). Used for the body-entry
        distance (d1_out) and to zero dose outside the body.
    model : a torch ``nn.Module`` already loaded with weights and ``.eval()``.
    device : torch device ('cuda'/'cpu').
    channels : tuple of channel names fed to the model, in order
        (default ('ct','d1','d1_out'); add 'd2'/'proj_mask' for other models).
    unscale : multiply the model output to get Gy (training used y/0.9*10, so
        unscale = 0.9/10 = 0.09).
    view_size_mm, out_size : ray-patch physical size (depth,width,height) and
        voxel size (Nx,Ny,Nz). Must match training.
    couch_deg, collimator_deg : DEPRECATED / ignored. The beam frame now always
        uses each beam's own gantry/couch/collimator from beam_geom (collimator=0
        is the identity, i.e. the no-rotation case).
    ct_norm : CT channel normalization, one of CT_NORMALIZERS ('hu' scaled HU,
        'density' relative electron density). Use 'density' for '..._ct_density'
        checkpoints. Overrides norm['ct'] when given.
    batch_size : beamlets predicted per forward pass.
    zero_outside_body : zero predicted dose outside the BODY mask.
    """

    def __init__(self, ct, body_mask_zyx, model, device="cpu",
                 channels=DEFAULT_CHANNELS, norm=None, ct_norm="hu", unscale=0.09,
                 view_size_mm=(512.0, 64.0, 64.0), out_size=(256, 32, 32),
                 couch_deg=0.0, collimator_deg=None, batch_size=8,
                 zero_outside_body=True, body_entry_step_mm=2.0, backend="gpu",
                 col_threshold_frac=0.0, resample_mode='linear'):
        if sitk is None:
            raise ImportError("SimpleITK is required for RayUNetPredictor.")
        import torch  # local import so portpy import does not require torch
        self._torch = torch
        self.ct = ct
        self.model = model
        self.device = device
        # per-beamlet relative threshold: drop dose < col_threshold_frac * column
        # peak (removes the model's low-level in-body "fog"; mirrors commercial TPS trunc)
        self.col_threshold_frac = float(col_threshold_frac)
        # ray-patch -> opt-voxel sampling: 'linear' (GPU grid_sample) or 'bspline'
        # (clamped cubic spline at the opt-voxel points via map_coordinates,
        # order=3; recovers the ~5% sub-grid peak linear averages away).
        self.resample_mode = str(resample_mode).lower()
        self.channels = tuple(channels)
        self.norm = dict(DEFAULT_NORM)
        if ct_norm is not None:
            if ct_norm not in CT_NORMALIZERS:
                raise ValueError("ct_norm must be one of %s" % list(CT_NORMALIZERS))
            self.norm["ct"] = CT_NORMALIZERS[ct_norm]
        if norm:
            self.norm.update(norm)
        self.unscale = float(unscale)
        self.view_size_mm = tuple(float(v) for v in view_size_mm)
        self.out_size = tuple(int(v) for v in out_size)
        self.couch_deg = couch_deg
        self.collimator_deg = collimator_deg          # value, or 'beam'
        self.batch_size = int(batch_size)
        self.zero_outside_body = zero_outside_body
        self.body_entry_step_mm = float(body_entry_step_mm)
        # 'gpu' -> torch.grid_sample resampling (fast); 'cpu' -> SimpleITK (reference)
        self.backend = "gpu" if str(backend).lower() == "gpu" else "cpu"

        # SimpleITK references (built once)
        self.ct_sitk = get_ct_sitk_image(ct)
        body = np.asarray(body_mask_zyx).astype(np.uint8)
        self.body_sitk = sitk.GetImageFromArray(body)
        self.body_sitk.CopyInformation(self.ct_sitk)
        self._body_np = body                                  # cached for ray-march

        # CT geometry shared by both backends (SimpleITK convention)
        self._ct_geom = (np.asarray(self.ct_sitk.GetOrigin(), float),
                         np.asarray(self.ct_sitk.GetSpacing(), float),
                         np.asarray(self.ct_sitk.GetDirection(), float).reshape(3, 3),
                         np.asarray(self.ct_sitk.GetSize(), int))

        # GPU backend: upload CT + body volumes to the device once (amortized).
        self._ct_gpu = self._body_gpu = None
        if self.backend == "gpu":
            ct_np = sitk.GetArrayFromImage(self.ct_sitk).astype(np.float32)
            self._ct_gpu = torch.from_numpy(ct_np).to(self.device)
            self._body_gpu = torch.from_numpy(body.astype(np.float32)).to(self.device)

    # ---- per-beamlet input construction -----------------------------------
    def _resample_ct_body(self, origin, spacing, D):
        """Resample CT + BODY onto the ray patch. grid_sample (gpu) or sitk (cpu)."""
        out = self.out_size
        if self.backend == "gpu":
            cO, cS, cDir, cN = self._ct_geom
            ct_zyx = ct_to_ray_patch(self._ct_gpu, cO, cS, cDir, cN,
                                     origin, spacing, D, out,
                                     default=-1000.0, mode="bilinear").cpu().numpy()
            body_zyx = (ct_to_ray_patch(self._body_gpu, cO, cS, cDir, cN,
                                        origin, spacing, D, out,
                                        default=0.0, mode="nearest").cpu().numpy()
                        > 0.5).astype(np.uint8)
        else:
            ct_zyx = sitk.GetArrayFromImage(resample_to_grid(
                self.ct_sitk, origin, spacing, D, out,
                default_value=-1000.0, interp=sitk.sitkLinear)).astype(np.float32)
            body_zyx = (sitk.GetArrayFromImage(resample_to_grid(
                self.body_sitk, origin, spacing, D, out,
                default_value=0.0, interp=sitk.sitkNearestNeighbor)) > 0.5).astype(np.uint8)
        return ct_zyx.astype(np.float32), body_zyx

    def _build_input(self, source_lps, center_lps, R):
        view, out = self.view_size_mm, self.out_size
        sx = view[0] / out[0]
        den = float(np.linalg.norm(np.asarray(center_lps) - np.asarray(source_lps)))
        # body-entry distance for d1_out / d1_in (cached body array -> no per-call copy)
        s_enter = estimate_body_entry_distance_mm(
            self.body_sitk, source_lps, center_lps,
            step_mm=self.body_entry_step_mm, mask_zyx=self._body_np)
        # put the isocenter plane at the middle of the patch along the ray
        x0_mm = max(0.0, den - (out[0] // 2) * sx)

        origin, spacing, D = build_ray_grid_with_beam_axes(
            source_lps, center_lps, R, view_size_mm=view, out_size=out, x0_mm=x0_mm)

        ct_zyx, body_zyx = self._resample_ct_body(origin, spacing, D)

        d1, d2, d1_out, _ = compute_ray_patch_d1_d2_split(
            body_zyx, s_enter, view_size_mm=view, out_size=out, x0_mm=x0_mm)

        raw = {"ct": ct_zyx, "d1": d1, "d2": d2, "d1_out": d1_out,
               "body": body_zyx.astype(np.float32)}
        chans = [self.norm[name](raw[name]) for name in self.channels]
        x = np.stack(chans, axis=0).astype(np.float32)       # (C, Z, Y, X)
        return x, (origin, spacing, D), body_zyx

    def _backproject(self, pred_zyx, meta):
        """Resample a ray-patch prediction onto the full CT grid (SimpleITK)."""
        origin, spacing, D = meta
        img = sitk.GetImageFromArray(np.asarray(pred_zyx, np.float32))
        img.SetOrigin(tuple(float(v) for v in np.asarray(origin).ravel()))
        img.SetSpacing(tuple(float(v) for v in np.asarray(spacing).ravel()))
        img.SetDirection(tuple(float(v) for v in np.asarray(D).reshape(-1)))
        res = sitk.ResampleImageFilter()
        res.SetReferenceImage(self.ct_sitk)
        res.SetInterpolator(sitk.sitkLinear)
        res.SetDefaultPixelValue(0.0)
        return sitk.GetArrayFromImage(res.Execute(img)).astype(np.float32)

    # ---- shared per-beam / per-batch machinery ----------------------------
    def _beam_frame(self, beam_geom):
        gantry = float(beam_geom["gantry_angle"])
        couch = float(beam_geom.get("couch_angle", 0.0))
        coll = float(beam_geom.get("collimator_angle", 0.0))
        R = get_rotation_matrix(gantry, couch, coll)
        iso = np.asarray(beam_geom["iso_xyz_mm"], float)
        source = source_lps_from_beam(iso, R, float(beam_geom["SAD_mm"]))
        return R, iso, source

    def _predict_batch(self, source, R, iso, bx, by, start, end):
        """Forward one batch of beamlets. Returns (idxs, preds_on_device, metas, bodies).

        preds is (B, Z, Y, X) torch on self.device, already * unscale and ReLU'd.
        """
        torch = self._torch
        xs, metas, bodies, idxs = [], [], [], []
        for k in range(start, end):
            center = beamlet_center_lps(bx[k], by[k], R, iso)
            x, meta, body_zyx = self._build_input(source, center, R)
            xs.append(x); metas.append(meta); bodies.append(body_zyx); idxs.append(k)
        xb = torch.from_numpy(np.stack(xs, axis=0)).to(self.device)
        with torch.no_grad():
            preds = self.model(xb)[:, 0] * self.unscale       # (B,Z,Y,X)
        preds = torch.clamp(preds, min=0.0)
        return idxs, preds, metas, bodies

    def _mask_body(self, p, body_zyx):
        """ReLU + zero-outside-body, on the same device as p (torch tensor)."""
        if self.zero_outside_body:
            torch = self._torch
            m = torch.as_tensor(body_zyx > 0.5, dtype=p.dtype, device=p.device)
            p = p * m
        return p

    # ---- main interface ---------------------------------------------------
    def predict_beam(self, beam_geom, beamlets):
        """Yield (local_idx, CT-grid 3D dose). Full-volume back-projection (SimpleITK)."""
        R, iso, source = self._beam_frame(beam_geom)
        bx = np.asarray(beamlets["position_x_mm"]).ravel()
        by = np.asarray(beamlets["position_y_mm"]).ravel()
        n = bx.size
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idxs, preds, metas, bodies = self._predict_batch(source, R, iso, bx, by, start, end)
            for j, k in enumerate(idxs):
                p = self._mask_body(preds[j], bodies[j])
                dose_ct = self._backproject(p.detach().cpu().numpy().astype(np.float32), metas[j])
                dose_ct[dose_ct < 0] = 0.0
                if self.col_threshold_frac > 0.0 and dose_ct.max() > 0:
                    dose_ct[dose_ct < self.col_threshold_frac * dose_ct.max()] = 0.0
                yield k, dose_ct

    def predict_beam_rows(self, beam_geom, beamlets, target_xyz, tol=0.0):
        """
        Direct path for the influence matrix: sample each beamlet's predicted dose
        at the voxel-row coordinates ``target_xyz`` (P, 3) and yield only the
        nonzero rows. Avoids the full-CT-grid back-projection entirely.

        Yields ``(local_idx, row_indices, row_doses)`` (numpy arrays).
        """
        torch = self._torch
        R, iso, source = self._beam_frame(beam_geom)
        cubic = (self.resample_mode == 'bspline')
        if cubic:
            from scipy.ndimage import map_coordinates
            txyz = np.asarray(target_xyz, np.float64)        # opt-voxel world coords
        else:
            tgt = torch.as_tensor(np.asarray(target_xyz, np.float32), device=self.device)
        bx = np.asarray(beamlets["position_x_mm"]).ravel()
        by = np.asarray(beamlets["position_y_mm"]).ravel()
        n = bx.size
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idxs, preds, metas, bodies = self._predict_batch(source, R, iso, bx, by, start, end)
            for j, k in enumerate(idxs):
                p = self._mask_body(preds[j], bodies[j])
                origin, spacing, D = metas[j]
                if cubic:
                    # cubic spline at the opt-voxel points: world -> ray-patch index
                    # (ix,iy,iz) = ((world-O) @ inv(D).T)/S ; array is (Nz,Ny,Nx).
                    pnp = p.detach().cpu().numpy()
                    O = np.asarray(origin, float); S = np.asarray(spacing, float)
                    DinvT = np.linalg.inv(np.asarray(D, float).reshape(3, 3)).T
                    ix = ((txyz - O) @ DinvT) / S            # (P,3) = (ix,iy,iz)
                    coords = np.stack([ix[:, 2], ix[:, 1], ix[:, 0]], axis=0)
                    vals = map_coordinates(pnp, coords, order=3,
                                           mode='constant', cval=0.0).astype(np.float32)
                    vals = np.clip(vals, 0.0, float(pnp.max())).astype(np.float32)
                else:
                    vals = sample_volume_at_world(p, tgt, origin, spacing, D,
                                                  self.out_size, default=0.0)
                    vals = vals.detach().cpu().numpy().astype(np.float32)
                if self.col_threshold_frac > 0.0 and vals.size:
                    vals[vals < self.col_threshold_frac * vals.max()] = 0.0
                nz = np.nonzero(vals > tol)[0]
                yield k, nz, vals[nz]


DEFAULT_BODY_NAMES = ("BODY", "PATIENT SURFACE", "EXTERNAL")


def _resolve_body_mask(structs, body_struct):
    """Return the BODY mask, trying each candidate name (str or list) in order."""
    names = list(structs.structures_dict.get('name', []))
    candidates = [body_struct] if isinstance(body_struct, str) else list(body_struct)
    for nm in candidates:
        if nm in names:
            return structs.get_structure_mask_3d(nm)
    raise ValueError("No BODY structure found. Tried %s; available: %s"
                     % (candidates, names))


def build_ray_unet_predictor(ct, structs, ckpt_path, device=None,
                             model_type='unet3d',
                             in_channels=3, num_levels=3, base_features=32,
                             num_groups=8, use_cbam=False,
                             body_struct=DEFAULT_BODY_NAMES,
                             **predictor_kwargs):
    """
    Convenience factory: load a beamlet-dose model checkpoint and wrap it in a
    RayUNetPredictor ready for ``InfluenceMatrix(..., predictor=...)``.

    model_type : 'attention_resunet3d' (default) or 'unet3d'.
    ``body_struct`` may be a name or a list of candidate names (first match wins);
    defaults to ('BODY','PATIENT SURFACE','EXTERNAL').
    ``predictor_kwargs`` are forwarded to ``RayUNetPredictor`` (channels, unscale,
    view_size_mm, out_size, batch_size, couch_deg, collimator_deg, ...).
    """
    import torch
    from ..models.attention_resunet3d import load_checkpoint

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == 'attention_resunet3d':
        from ..models.attention_resunet3d import AttentionResUNet3D
        model = AttentionResUNet3D(in_channels=in_channels, out_channels=1,
                                   base_features=base_features, num_groups=num_groups,
                                   use_cbam=use_cbam, num_levels=num_levels).to(device)
    else:
        from ..models.unet3d import UNet3D
        model = UNet3D(in_channels=in_channels, out_channels=1,
                       base_features=base_features, num_groups=num_groups).to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    body_zyx = _resolve_body_mask(structs, body_struct)
    return RayUNetPredictor(ct=ct, body_mask_zyx=body_zyx, model=model,
                            device=device, **predictor_kwargs)


# ===========================================================================
# Patient-grid predictor (fix_sampling approach)
# ===========================================================================
class PatientGridPredictor(BeamletDosePredictor):
    """
    Patient-grid (fixed isotropic box) beamlet-dose predictor.

    Instead of a ray-aligned BEV patch this predictor works on a fixed
    isotropic patient-centred grid (default 2.5 mm, 224×192×96).  The full
    patient cross-section is covered so scatter dose to distant OARs is
    predicted correctly.

    Architecture / normalisation (must match the fix_sampling checkpoint):
      model      : UNet3D(in_channels=4)
      channels   : [ct_norm, d2_norm, d1_norm, d1_out_norm]
      ct_norm    : (clip(HU,-1000,3071)+1000)/4071  (scaled HU, NOT density)
      d1_norm    : d1 / 1600
      d2_norm    : d2 / 600
      d1_out_norm: d1_out / 800   (d1_out = body_mask * t_entry scalar)
      unscale    : 0.07  (training: y_norm = y / 0.7 * 10)
    """

    def __init__(self, ct, body_mask_zyx, model, device="cpu",
                 center_xyz_mm=None,
                 spacing_xyz_mm=(2.5, 2.5, 2.5),
                 size_xyz=(224, 192, 96),
                 unscale=0.07,
                 batch_size=4,
                 zero_outside_body=True,
                 ray_radius_mm=3.0,
                 col_threshold_frac=0.0,
                 resample_mode='linear'):
        import torch
        self._torch = torch
        self.ct = ct
        self.model = model
        self.device = device
        self.unscale = float(unscale)
        self.batch_size = int(batch_size)
        self.zero_outside_body = zero_outside_body
        # per-beamlet relative threshold (drop dose < frac * column peak)
        self.col_threshold_frac = float(col_threshold_frac)
        self.ray_radius_mm = float(ray_radius_mm)
        # patient-grid -> opt-voxel sampling: 'linear' (GPU grid_sample) or
        # 'bspline' (clamped cubic). For 'bspline', predict_beam_rows evaluates a
        # cubic spline DIRECTLY at the opt-voxel points via map_coordinates
        # (order=3) -- fast, no full-CT-grid resample -- so it stays on the rows
        # path (predict_beam, the 3D API, uses SimpleITK B-spline for its grid).
        self.resample_mode = str(resample_mode).lower()
        self.force_full_backprojection = False

        from .patient_grid_geometry import (
            make_reference_image_from_center,
            resample_ct_to_patient_grid,
            resample_mask_to_patient_grid,
            resample_pred_to_ct_grid,
            precompute_beam_geometry_on_patient_grid,
            _ct_sitk_from_portpy,
        )

        ref_img, self._ref_meta = make_reference_image_from_center(
            center_xyz_mm=center_xyz_mm, spacing_xyz_mm=spacing_xyz_mm, size_xyz=size_xyz)

        # CT and body mask resampled to patient grid, uploaded to GPU once
        ct_patient = resample_ct_to_patient_grid(ct, ref_img)
        self._ct_gpu = torch.from_numpy(
            np.clip((ct_patient + 1000.0) / 4071.0, 0.0, 1.0).astype(np.float32)
        ).to(device)                                                      # (Nz, Ny, Nx)

        body_patient = resample_mask_to_patient_grid(
            np.asarray(body_mask_zyx, np.uint8), ct, ref_img)
        self._body_gpu = torch.from_numpy(
            body_patient.astype(np.float32)).to(device) > 0.5            # bool (Nz, Ny, Nx)

        self._ct_sitk        = _ct_sitk_from_portpy(ct)
        self._resample_back  = resample_pred_to_ct_grid
        self._precompute_geom = precompute_beam_geometry_on_patient_grid
        self._patient_size_xyz = tuple(int(v) for v in size_xyz)
        self._patient_origin   = np.asarray(self._ref_meta["origin_xyz_mm"],  float)
        self._patient_spacing  = np.asarray(self._ref_meta["spacing_xyz_mm"], float)
        self._patient_dir      = np.eye(3, dtype=float)

    # ---- batch: d-channels computed on GPU, different per beamlet --------
    def _build_input_batch(self, geom_gpu, bx, by, SAD_mm, start, end):
        """
        d1, d2, d1_out are computed on GPU for each beamlet in [start, end).
        They differ per beamlet because ux/uy/uz = f(b_x[k], b_y[k]) changes
        per beamlet; rx/ry/rz (beam-frame voxel positions) are shared within
        the beam and broadcast over the batch dimension.

        Returns (list_of_local_idxs, preds (B, Nz, Ny, Nx) on device).
        """
        torch = self._torch
        B = end - start

        # Per-beamlet ray direction unit vectors (B, 1, 1, 1) for broadcasting
        bx_t = torch.tensor(bx[start:end], dtype=torch.float32, device=self.device)
        by_t = torch.tensor(by[start:end], dtype=torch.float32, device=self.device)
        den  = (bx_t**2 + float(SAD_mm)**2 + by_t**2).sqrt().clamp(min=1e-12)
        ux = (bx_t / den).view(B, 1, 1, 1)
        uy = (float(SAD_mm) / den).view(B, 1, 1, 1)
        uz = (by_t / den).view(B, 1, 1, 1)

        # d1/d2: (B, Nz, Ny, Nx) -- different for each beamlet via broadcasting
        rx, ry, rz, r2 = geom_gpu["rx"], geom_gpu["ry"], geom_gpu["rz"], geom_gpu["r2"]
        d1 = rx * ux + ry * uy + rz * uz          # along-ray distance from source
        d2 = (r2 - d1 * d1).clamp(min=0.0).sqrt() # perpendicular distance from ray

        # d1_out: body_mask * t_entry, t_entry = min d1 near ray inside body
        body       = self._body_gpu.unsqueeze(0)   # (1, Nz, Ny, Nx)
        inside_ray = body & (d2 <= self.ray_radius_mm)
        t_entry    = d1.masked_fill(~inside_ray, float('inf')).view(B, -1).min(dim=1).values
        t_entry    = t_entry.nan_to_num(nan=0.0, posinf=0.0)   # no-hit → 0
        d1_out     = body.float() * t_entry.view(B, 1, 1, 1)   # (B, Nz, Ny, Nx)

        # NOTE: distance channels are NOT clamped -- training divides only
        # (d2/600, d1/1600, d1_out/800) and these exceed 1.0 on the patient grid.
        xb = torch.stack([
            self._ct_gpu.unsqueeze(0).expand(B, -1, -1, -1),
            d2     /  600.0,
            d1     / 1600.0,
            d1_out /  800.0,
        ], dim=1)                                  # (B, 4, Nz, Ny, Nx)

        with torch.no_grad():
            preds = self.model(xb)[:, 0] * self.unscale
        preds = torch.clamp(preds, min=0.0)
        if self.zero_outside_body:
            preds = preds * body.float()
        return list(range(start, end)), preds

    def _resample_pred(self, pred_np):
        """Patient grid -> CT grid. 'linear', or 'bspline' = clamped cubic.

        Clamped cubic recovers the sub-grid peak that linear averages away, but
        clamps to the local bracketing-cell max (NN of a 2x2x2 max-filter) so the
        cubic ringing/overshoot can't create dose above the source data; floor 0.
        """
        if self.resample_mode == 'bspline':
            from scipy.ndimage import maximum_filter
            dose_cubic = self._resample_back(pred_np, self._ref_meta, self._ct_sitk,
                                             interp=sitk.sitkBSpline)
            local_max = maximum_filter(pred_np, size=2)
            dose_hi = self._resample_back(local_max, self._ref_meta, self._ct_sitk,
                                          interp=sitk.sitkNearestNeighbor)
            return np.clip(dose_cubic, 0.0, dose_hi)
        return self._resample_back(pred_np, self._ref_meta, self._ct_sitk)

    # ---- public API -------------------------------------------------------
    def predict_beam(self, beam_geom, beamlets):
        """Yield (local_idx, CT-grid 3D dose). Full back-projection via SimpleITK."""
        bx     = np.asarray(beamlets["position_x_mm"]).ravel()
        by     = np.asarray(beamlets["position_y_mm"]).ravel()
        iso    = np.asarray(beam_geom["iso_xyz_mm"], float)
        SAD_mm = float(beam_geom["SAD_mm"])
        # precompute beam geometry on CPU once per beam, upload to GPU
        cpu_geom = self._precompute_geom(self._ref_meta, iso,
                                         float(beam_geom["gantry_angle"]),
                                         float(beam_geom.get("couch_angle", 0.0)),
                                         beam_geom.get("collimator_angle", 0.0), SAD_mm)
        geom_gpu = {k: self._torch.from_numpy(v).to(self.device) for k, v in cpu_geom.items()}

        n = bx.size
        for start in range(0, n, self.batch_size):
            idxs, preds = self._build_input_batch(
                geom_gpu, bx, by, SAD_mm, start, min(start + self.batch_size, n))
            for j, k in enumerate(idxs):
                pred_np = preds[j].detach().cpu().numpy()
                dose_ct = self._resample_pred(pred_np)
                dose_ct[dose_ct < 0] = 0.0
                if self.col_threshold_frac > 0.0 and dose_ct.max() > 0:
                    dose_ct[dose_ct < self.col_threshold_frac * dose_ct.max()] = 0.0
                yield k, dose_ct

    def predict_beam_rows(self, beam_geom, beamlets, target_xyz, tol=0.0):
        """Sample predictions at voxel-row world coordinates. Yields (idx, rows, vals).

        'linear' -> GPU grid_sample. 'bspline' -> clamped cubic spline evaluated
        directly at the opt-voxel points via scipy map_coordinates(order=3) (no
        full-CT-grid resample), clamped to the local 2x2x2 max to bound ringing.
        """
        from .gpu_resample import sample_volume_at_world
        torch  = self._torch
        bx     = np.asarray(beamlets["position_x_mm"]).ravel()
        by     = np.asarray(beamlets["position_y_mm"]).ravel()
        iso    = np.asarray(beam_geom["iso_xyz_mm"], float)
        SAD_mm = float(beam_geom["SAD_mm"])
        # precompute beam geometry on CPU once per beam, upload to GPU
        cpu_geom = self._precompute_geom(self._ref_meta, iso,
                                         float(beam_geom["gantry_angle"]),
                                         float(beam_geom.get("couch_angle", 0.0)),
                                         beam_geom.get("collimator_angle", 0.0), SAD_mm)
        geom_gpu = {k: torch.from_numpy(v).to(self.device) for k, v in cpu_geom.items()}

        cubic = (self.resample_mode == 'bspline')
        if cubic:
            from scipy.ndimage import map_coordinates
            # opt-voxel world coords -> patient-grid continuous index (z,y,x), once/beam
            txyz = np.asarray(target_xyz, np.float64)
            o, s = self._patient_origin, self._patient_spacing
            self._idx_zyx = np.stack([(txyz[:, 2] - o[2]) / s[2],
                                      (txyz[:, 1] - o[1]) / s[1],
                                      (txyz[:, 0] - o[0]) / s[0]], axis=0)
            # GPU cubic (cupy) is ~88x faster than CPU scipy and numerically identical;
            # fall back to scipy when cupy is unavailable or force_cpu_cubic is set.
            self._cnd = None
            if not getattr(self, 'force_cpu_cubic', False):
                try:
                    import cupy as _cp
                    import cupyx.scipy.ndimage as _cnd
                    self._cp, self._cnd = _cp, _cnd
                    self._idx_zyx_cp = _cp.asarray(self._idx_zyx)
                except Exception:
                    self._cnd = None
        else:
            tgt = torch.as_tensor(np.asarray(target_xyz, np.float32), device=self.device)

        n = bx.size
        for start in range(0, n, self.batch_size):
            idxs, preds = self._build_input_batch(
                geom_gpu, bx, by, SAD_mm, start, min(start + self.batch_size, n))
            for j, k in enumerate(idxs):
                if cubic:
                    if self._cnd is not None:
                        # GPU cubic: share the torch pred via DLPack (no CPU roundtrip),
                        # cubic-interpolate on GPU, clamp to [0, grid max], pull only the values.
                        p_cp = self._cp.from_dlpack(preds[j].contiguous())
                        vals_cp = self._cnd.map_coordinates(p_cp, self._idx_zyx_cp, order=3,
                                                            mode='constant', cval=0.0)
                        vals = self._cp.asnumpy(
                            self._cp.clip(vals_cp, 0.0, float(p_cp.max()))).astype(np.float32)
                    else:
                        p = preds[j].detach().cpu().numpy()
                        vals = map_coordinates(p, self._idx_zyx, order=3,
                                               mode='constant', cval=0.0).astype(np.float32)
                        # clamp to [0, per-beamlet grid max]: bounds cubic overshoot so
                        # no hotspot above the source peak (smooth UNet output -> negligible
                        # penumbra ringing).
                        vals = np.clip(vals, 0.0, float(p.max())).astype(np.float32)
                else:
                    vals = sample_volume_at_world(
                        preds[j], tgt,
                        self._patient_origin, self._patient_spacing,
                        self._patient_dir, self._patient_size_xyz, default=0.0)
                    vals = vals.detach().cpu().numpy().astype(np.float32)
                if self.col_threshold_frac > 0.0 and vals.size:
                    vals[vals < self.col_threshold_frac * vals.max()] = 0.0
                nz = np.nonzero(vals > tol)[0]
                yield k, nz, vals[nz]


def build_patient_grid_predictor(ct, structs, ckpt_path, device=None,
                                  in_channels=4, base_features=32, num_groups=8,
                                  spacing_xyz_mm=(2.5, 2.5, 2.5),
                                  size_xyz=(224, 192, 96),
                                  unscale=0.07,
                                  batch_size=4,
                                  body_struct=DEFAULT_BODY_NAMES,
                                  iso_xyz_mm=None,
                                  **predictor_kwargs):
    """
    Load a fix_sampling UNet3D(in_channels=4) checkpoint and wrap it in a
    PatientGridPredictor ready for InfluenceMatrix(..., predictor=...).

    iso_xyz_mm : isocenter (x,y,z mm) used to centre the patient grid in Z.
                 If None, uses the CT mid-plane z.  For standard IMRT all
                 beams share one isocenter so passing it here is preferred.
    """
    import torch
    from ..models.attention_resunet3d import load_checkpoint
    from ..models.unet3d import UNet3D
    from .patient_grid_geometry import get_body_center_xy_mm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet3D(in_channels=in_channels, out_channels=1,
                   base_features=base_features, num_groups=num_groups).to(device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    body_zyx = _resolve_body_mask(structs, body_struct)

    # Determine patient grid centre: body XY + isocenter Z
    body_cx, body_cy = get_body_center_xy_mm(structs, ct)
    if iso_xyz_mm is not None:
        center_z = float(iso_xyz_mm[2])
    else:
        ct_arr = ct.ct_dict['ct_hu_3d']
        if isinstance(ct_arr, (list, tuple)):
            ct_arr = ct_arr[0]
        origin_z  = float(ct.ct_dict['origin_xyz_mm'][2])
        spacing_z = float(ct.ct_dict['resolution_xyz_mm'][2])
        center_z  = origin_z + (ct_arr.shape[0] - 1) / 2.0 * spacing_z

    center_xyz = (body_cx, body_cy, center_z)

    return PatientGridPredictor(
        ct=ct, body_mask_zyx=body_zyx, model=model, device=device,
        center_xyz_mm=center_xyz,
        spacing_xyz_mm=spacing_xyz_mm,
        size_xyz=size_xyz,
        unscale=unscale,
        batch_size=batch_size,
        **predictor_kwargs
    )