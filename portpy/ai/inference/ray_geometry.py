# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# ----------------------------------------------------------------------
"""
Ray-aligned beam geometry helpers for beamlet-dose prediction.

These are vendored (self-contained) from the validated research utilities
(``portpy_ai_utils`` + ``ANet_preprocess4_only_ct_linux``) so the production
inference workflow has no dependency on external scripts. Everything here is
pure geometry: building the per-beamlet ray-aligned patch grid, resampling the
CT/body onto it, and computing the along-ray / perpendicular distance channels.

Conventions (must match training):
  * LPS patient frame. Row-vector rotation: ``r_beam = r_lps @ R`` where
    ``R = get_rotation_matrix(gantry, couch, collimator)``.
  * Beam axis is +y' in the beam frame; the source sits at beam-frame
    ``(0, -SAD, 0)``.
  * A ray-aligned patch has axes: X = source->beamlet-center ray (depth),
    Y/Z = lateral. Arrays returned in Z,Y,X order (SimpleITK / numpy convention).
"""
import numpy as np

try:
    import SimpleITK as sitk
except Exception:                       # pragma: no cover - sitk optional at import
    sitk = None


def get_rotation_matrix(gantry_deg: float, couch_deg: float = 0.0,
                        collimator_deg: float = 0.0) -> np.ndarray:
    """Patient(LPS)->beam rotation, row-vector convention (``r_beam = r_lps @ R``)."""
    g = np.deg2rad(gantry_deg)
    c = np.deg2rad(couch_deg)
    Rg = np.array([[np.cos(g), -np.sin(g), 0.0],
                   [np.sin(g),  np.cos(g), 0.0],
                   [0.0,        0.0,       1.0]], float)
    Rc = np.array([[np.cos(c), 0.0, np.sin(c)],
                   [0.0,       1.0, 0.0],
                   [-np.sin(c), 0.0, np.cos(c)]], float)
    rot = Rc @ Rg
    t = np.deg2rad(collimator_deg)
    # Collimator rotates the field about the BEAM CENTRAL AXIS (beam-frame y),
    # not about z. About-z mis-rotates off-axis beamlets and shifts the source
    # off the central axis; about-y leaves the source at (0,-SAD,0) fixed.
    # collimator_deg=0 -> identity (the no-collimator case).
    r_coll = np.array([[np.cos(t), 0.0, -np.sin(t)],
                       [0.0,       1.0,  0.0],
                       [np.sin(t), 0.0,  np.cos(t)]], float)
    return rot @ r_coll


def get_ct_sitk_image(ct):
    """Build a SimpleITK image from a PortPy CT object (HU, Z,Y,X array)."""
    if sitk is None:
        raise ImportError("SimpleITK is required for ray-aligned resampling.")
    ct_arr = ct.ct_dict['ct_hu_3d']
    if isinstance(ct_arr, (list, tuple)):      # PortPy wraps as [arr]; ct_test stores arr
        ct_arr = ct_arr[0]
    img = sitk.GetImageFromArray(np.asarray(ct_arr))
    img.SetOrigin(tuple(float(v) for v in ct.ct_dict['origin_xyz_mm']))
    img.SetSpacing(tuple(float(v) for v in ct.ct_dict['resolution_xyz_mm']))
    img.SetDirection(tuple(float(v) for v in ct.ct_dict['direction']))
    return img


def resample_to_grid(img_sitk, origin, spacing, direction_3x3, size_xyz,
                     default_value=0.0, interp=None):
    """Resample a SimpleITK image onto an explicit (origin, spacing, direction) grid."""
    if sitk is None:
        raise ImportError("SimpleITK is required for ray-aligned resampling.")
    if interp is None:
        interp = sitk.sitkLinear
    Nx, Ny, Nz = (int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2]))
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(img_sitk)
    res.SetOutputOrigin(tuple(float(v) for v in np.asarray(origin).ravel()))
    res.SetOutputSpacing(tuple(float(v) for v in np.asarray(spacing).ravel()))
    res.SetOutputDirection(tuple(float(v) for v in np.asarray(direction_3x3).reshape(-1)))
    res.SetSize([Nx, Ny, Nz])
    res.SetInterpolator(interp)
    res.SetDefaultPixelValue(float(default_value))
    res.SetOutputPixelType(sitk.sitkFloat32)
    return res.Execute(img_sitk)


def estimate_body_entry_distance_mm(body_mask_sitk, source_lps, beamlet_center_lps,
                                    max_dist_mm: float = 2000.0, step_mm: float = 2.0,
                                    mask_zyx=None):
    """
    Distance (mm) from the source to where the central ray first enters BODY.
    Returns None if the ray never hits the body mask. (t_entry for d1_out/d1_in.)

    ``mask_zyx`` may be a precomputed array of ``body_mask_sitk`` to avoid a
    per-call ``GetArrayFromImage`` (important when looping over many beamlets).
    """
    src = np.asarray(source_lps, float)
    tgt = np.asarray(beamlet_center_lps, float)
    v = tgt - src
    den = float(np.linalg.norm(v))
    if den < 1e-6:
        return None
    u = v / den

    if mask_zyx is None:
        mask_zyx = sitk.GetArrayFromImage(body_mask_sitk)
    size_xyz = np.array(body_mask_sitk.GetSize(), dtype=int)
    origin = np.array(body_mask_sitk.GetOrigin(), dtype=float)
    spacing = np.array(body_mask_sitk.GetSpacing(), dtype=float)
    D = np.array(body_mask_sitk.GetDirection(), dtype=float).reshape(3, 3)

    s_max = min(max_dist_mm, den + 1000.0)
    s_vals = np.arange(0.0, s_max + 1e-6, step_mm, dtype=float)
    pts = src[None, :] + s_vals[:, None] * u[None, :]
    idx_f = ((pts - origin[None, :]) @ D) / spacing[None, :]
    idx = np.rint(idx_f).astype(int)
    inb = ((idx[:, 0] >= 0) & (idx[:, 0] < size_xyz[0]) &
           (idx[:, 1] >= 0) & (idx[:, 1] < size_xyz[1]) &
           (idx[:, 2] >= 0) & (idx[:, 2] < size_xyz[2]))
    if not np.any(inb):
        return None
    idx_in = idx[inb]
    s_in = s_vals[inb]
    hit = mask_zyx[idx_in[:, 2], idx_in[:, 1], idx_in[:, 0]] > 0
    if not np.any(hit):
        return None
    return float(s_in[int(np.flatnonzero(hit)[0])])


def build_ray_grid_with_beam_axes(source_lps, target_lps, R_lps_to_bcs,
                                  view_size_mm=(512.0, 64.0, 64.0),
                                  out_size=(256, 32, 32), x0_mm=0.0):
    """
    Build a ray-aligned patch grid for one beamlet.

    Axes: X = source->beamlet-center ray; Y = beam-frame x projected perpendicular
    to the ray; Z completes a right-handed frame (~beam-frame z). Returns
    (origin, spacing, D) where ``origin`` is the physical center of voxel (0,0,0),
    ``spacing`` = (sx,sy,sz) mm, ``D`` is the 3x3 SimpleITK direction (columns ux,uy,uz).
    """
    source_lps = np.asarray(source_lps, float)
    target_lps = np.asarray(target_lps, float)
    R = np.asarray(R_lps_to_bcs, float)

    v = target_lps - source_lps
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("source and target too close.")
    ux = v / n

    beam_x_lps = np.array([1.0, 0.0, 0.0]) @ R.T
    beam_z_lps = np.array([0.0, 0.0, 1.0]) @ R.T

    uy = beam_x_lps - np.dot(beam_x_lps, ux) * ux
    if np.linalg.norm(uy) < 1e-8:
        fallback = np.array([0.0, 0.0, 1.0])
        uy = fallback - np.dot(fallback, ux) * ux
    uy = uy / (np.linalg.norm(uy) + 1e-12)

    uz = np.cross(ux, uy)
    uz = uz / (np.linalg.norm(uz) + 1e-12)
    if np.dot(uz, beam_z_lps) < 0:        # keep uz roughly along beam-frame z
        uz = -uz
        uy = -uy

    depth_mm, width_mm, height_mm = map(float, view_size_mm)
    Nx, Ny, Nz = map(int, out_size)
    sx, sy, sz = depth_mm / Nx, width_mm / Ny, height_mm / Nz

    D = np.stack([ux, uy, uz], axis=1)
    origin = (source_lps
              + ux * (float(x0_mm) + 0.5 * sx)
              - uy * (width_mm / 2.0 - 0.5 * sy)
              - uz * (height_mm / 2.0 - 0.5 * sz))
    spacing = (sx, sy, sz)
    return origin, spacing, D


def compute_ray_patch_d1_d2_split(body_mask_zyx, t_entry,
                                  view_size_mm=(512.0, 64.0, 64.0),
                                  out_size=(256, 32, 32), x0_mm=0.0):
    """
    Distance channels in the ray-aligned patch (all Z,Y,X):
      d1     : distance from source along the beamlet ray (= local X).
      d2     : perpendicular distance from the ray (= sqrt(Y^2+Z^2)).
      d1_out : distance before body entry (min(d1, t_entry)).
      d1_in  : distance inside body after entry (0 outside BODY).
    """
    depth_mm, width_mm, height_mm = map(float, view_size_mm)
    Nx, Ny, Nz = map(int, out_size)
    sx, sy, sz = depth_mm / Nx, width_mm / Ny, height_mm / Nz

    x = float(x0_mm) + (np.arange(Nx, dtype=np.float32) + 0.5) * sx
    y = -0.5 * width_mm + (np.arange(Ny, dtype=np.float32) + 0.5) * sy
    z = -0.5 * height_mm + (np.arange(Nz, dtype=np.float32) + 0.5) * sz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    d1_xyz = X.astype(np.float32)
    d2_xyz = np.sqrt(Y * Y + Z * Z).astype(np.float32)
    d1_zyx = np.transpose(d1_xyz, (2, 1, 0))
    d2_zyx = np.transpose(d2_xyz, (2, 1, 0))

    body = body_mask_zyx.astype(bool)
    if t_entry is None:
        d1_out_zyx = d1_zyx.astype(np.float32).copy()
        d1_in_zyx = np.zeros_like(d1_zyx, dtype=np.float32)
    else:
        t_entry = float(t_entry)
        d1_out_zyx = np.minimum(d1_zyx, t_entry).astype(np.float32)
        d1_in_zyx = np.maximum(d1_zyx - t_entry, 0.0).astype(np.float32)
        d1_in_zyx[~body] = 0.0
    return d1_zyx, d2_zyx, d1_out_zyx, d1_in_zyx


def source_lps_from_beam(iso_xyz_mm, R_lps_to_bcs, SAD_mm):
    """Patient-LPS coordinates of the X-ray source (beam-frame (0,-SAD,0))."""
    src_bcs = np.array([0.0, -float(SAD_mm), 0.0], dtype=float)
    return (src_bcs @ np.asarray(R_lps_to_bcs, float).T) + np.asarray(iso_xyz_mm, float)


def beamlet_center_lps(b_x_mm, b_y_mm, R_lps_to_bcs, iso_xyz_mm):
    """Patient-LPS coordinates of a beamlet center (beam-frame (b_x, 0, b_y))."""
    c_bcs = np.array([float(b_x_mm), 0.0, float(b_y_mm)], dtype=float)
    return (c_bcs @ np.asarray(R_lps_to_bcs, float).T) + np.asarray(iso_xyz_mm, float)