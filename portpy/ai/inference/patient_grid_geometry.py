# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# ----------------------------------------------------------------------
"""
Patient-grid geometry helpers for PatientGridPredictor.

Vendored from portpy_ai_utils.py (fix_sampling preprocessing pipeline) so the
inference workflow is self-contained. All functions operate on a fixed
isotropic patient-centred box (e.g. 2.5 mm, 224×192×96 voxels).
"""
import numpy as np
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from .ray_geometry import get_rotation_matrix   # reuse existing


# --------------------------------------------------------------------------
# SimpleITK / numpy conversions
# --------------------------------------------------------------------------
def np_zyx_to_sitk(arr_zyx, origin_xyz, spacing_xyz, direction=None):
    if sitk is None:
        raise ImportError("SimpleITK is required for PatientGridPredictor.")
    img = sitk.GetImageFromArray(np.asarray(arr_zyx, np.float32))
    img.SetOrigin(tuple(float(v) for v in origin_xyz))
    img.SetSpacing(tuple(float(v) for v in spacing_xyz))
    if direction is None:
        direction = np.eye(3).reshape(-1).tolist()
    img.SetDirection(tuple(float(v) for v in direction))
    return img


def sitk_to_np_zyx(img_sitk):
    return sitk.GetArrayFromImage(img_sitk).astype(np.float32)


def resample_to_reference(img_sitk, ref_img, is_mask=False, default_value=0.0, interp=None):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref_img)
    if interp is not None:
        r.SetInterpolator(interp)
    else:
        r.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    r.SetDefaultPixelValue(float(default_value))
    r.SetOutputPixelType(sitk.sitkFloat32)
    return r.Execute(img_sitk)


# --------------------------------------------------------------------------
# Reference grid construction
# --------------------------------------------------------------------------
def make_reference_image_from_center(center_xyz_mm, spacing_xyz_mm=(2.5, 2.5, 2.5),
                                     size_xyz=(224, 192, 96), direction=None):
    """Fixed-size isotropic reference image centred at center_xyz_mm.

    size_xyz = (Nx, Ny, Nz). Returns (sitk.Image, meta_dict).
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for PatientGridPredictor.")
    center  = np.asarray(center_xyz_mm,  dtype=np.float64)
    spacing = np.asarray(spacing_xyz_mm, dtype=np.float64)
    size    = np.asarray(size_xyz,       dtype=np.int32)
    if direction is None:
        direction = np.eye(3, dtype=np.float64).reshape(-1).tolist()
    origin = center - spacing * (size - 1) / 2.0
    ref = sitk.Image([int(size[0]), int(size[1]), int(size[2])], sitk.sitkFloat32)
    ref.SetOrigin(tuple(origin))
    ref.SetSpacing(tuple(spacing))
    ref.SetDirection(tuple(direction))
    meta = {
        "origin_xyz_mm":  tuple(origin.tolist()),
        "spacing_xyz_mm": tuple(spacing.tolist()),
        "size_xyz":       tuple(size.tolist()),
        "center_xyz_mm":  tuple(center.tolist()),
    }
    return ref, meta


# --------------------------------------------------------------------------
# Body centre
# --------------------------------------------------------------------------
def get_body_center_xy_mm(structs, ct):
    """Mean X/Y of the BODY mask bounding box in world coordinates."""
    body_mask = structs.get_structure_mask_3d('BODY')   # (Nz, Ny, Nx)
    _, y_idx, x_idx = np.where(body_mask > 0)
    if len(x_idx) == 0:
        raise ValueError("BODY mask is empty.")
    spacing = np.asarray(ct.ct_dict['resolution_xyz_mm'], dtype=np.float64)
    origin  = np.asarray(ct.ct_dict['origin_xyz_mm'],     dtype=np.float64)
    return float(origin[0] + x_idx.mean() * spacing[0]), \
           float(origin[1] + y_idx.mean() * spacing[1])


# --------------------------------------------------------------------------
# CT / mask resampling to patient grid
# --------------------------------------------------------------------------
def _ct_sitk_from_portpy(ct):
    ct_arr = ct.ct_dict['ct_hu_3d']
    if isinstance(ct_arr, (list, tuple)):
        ct_arr = ct_arr[0]
    direction = ct.ct_dict.get('direction', np.eye(3).reshape(-1).tolist())
    return np_zyx_to_sitk(np.asarray(ct_arr, np.float32),
                           origin_xyz=ct.ct_dict['origin_xyz_mm'],
                           spacing_xyz=ct.ct_dict['resolution_xyz_mm'],
                           direction=direction)


def resample_ct_to_patient_grid(ct, ref_img, default_ct=-1000.0):
    """Resample PortPy CT onto the fixed patient reference grid → (Nz,Ny,Nx) array."""
    return sitk_to_np_zyx(resample_to_reference(
        _ct_sitk_from_portpy(ct), ref_img, is_mask=False, default_value=default_ct))


def resample_mask_to_patient_grid(mask_zyx, ct, ref_img):
    """Resample a CT-grid binary mask → (Nz,Ny,Nx) uint8 on the patient grid."""
    direction = ct.ct_dict.get('direction', np.eye(3).reshape(-1).tolist())
    img_sitk = np_zyx_to_sitk(mask_zyx.astype(np.float32),
                               origin_xyz=ct.ct_dict['origin_xyz_mm'],
                               spacing_xyz=ct.ct_dict['resolution_xyz_mm'],
                               direction=direction)
    out = sitk_to_np_zyx(resample_to_reference(img_sitk, ref_img, is_mask=True))
    return (out > 0.5).astype(np.uint8)


def resample_pred_to_ct_grid(pred_zyx, ref_meta, ct_sitk, default=0.0, interp=None):
    """Resample prediction from patient grid back to full CT grid → (Nz,Ny,Nx).

    interp : SimpleITK interpolator (e.g. sitk.sitkLinear, sitk.sitkBSpline,
             sitk.sitkNearestNeighbor). None -> linear.
    """
    pred_sitk = np_zyx_to_sitk(
        pred_zyx,
        origin_xyz=ref_meta["origin_xyz_mm"],
        spacing_xyz=ref_meta["spacing_xyz_mm"],
        direction=np.eye(3).reshape(-1).tolist(),
    )
    return sitk_to_np_zyx(resample_to_reference(
        pred_sitk, ct_sitk, is_mask=False, default_value=default, interp=interp))


# --------------------------------------------------------------------------
# Per-beam geometry (precomputed once per beam, reused for all beamlets)
# --------------------------------------------------------------------------
def precompute_beam_geometry_on_patient_grid(ref_meta, iso_xyz_mm,
                                             gantry, couch=0.0,
                                             collimator=0.0, SAD_mm=1000.0):
    """
    Compute beam-frame voxel positions on the patient grid.
    Returns dict with rx, ry, rz, r2 in (Nz, Ny, Nx) order, reused by
    compute_d1_d2_on_patient_grid for each beamlet.
    """
    origin  = np.asarray(ref_meta["origin_xyz_mm"],  dtype=np.float32)
    spacing = np.asarray(ref_meta["spacing_xyz_mm"], dtype=np.float32)
    size    = np.asarray(ref_meta["size_xyz"],        dtype=np.int32)   # (Nx, Ny, Nz)
    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])

    x = origin[0] + spacing[0] * np.arange(nx, dtype=np.float32)
    y = origin[1] + spacing[1] * np.arange(ny, dtype=np.float32)
    z = origin[2] + spacing[2] * np.arange(nz, dtype=np.float32)
    Zg, Yg, Xg = np.meshgrid(z, y, x, indexing='ij')   # (Nz, Ny, Nx)

    iso = np.asarray(iso_xyz_mm, dtype=np.float32)
    R   = get_rotation_matrix(gantry, couch, collimator).astype(np.float32)
    pts = np.stack([(Xg - iso[0]).ravel(),
                    (Yg - iso[1]).ravel(),
                    (Zg - iso[2]).ravel()], axis=1)     # (N, 3)
    rb  = pts @ R
    xp  = rb[:, 0].reshape(Zg.shape)
    yp  = rb[:, 1].reshape(Zg.shape)
    zp  = rb[:, 2].reshape(Zg.shape)

    ry = (yp + float(SAD_mm)).astype(np.float32)
    rx = xp.astype(np.float32)
    rz = zp.astype(np.float32)
    return {"rx": rx, "ry": ry, "rz": rz,
            "r2": (rx*rx + ry*ry + rz*rz).astype(np.float32)}


# --------------------------------------------------------------------------
# Per-beamlet distance channels on the patient grid
# --------------------------------------------------------------------------
def compute_d1_d2_on_patient_grid(beam_geom, b_x, b_y, SAD_mm, eps=1e-12):
    """d1 (along-ray distance from source) and d2 (perpendicular) on patient grid."""
    rx, ry, rz, r2 = beam_geom["rx"], beam_geom["ry"], beam_geom["rz"], beam_geom["r2"]
    den = float(np.sqrt(b_x*b_x + SAD_mm*SAD_mm + b_y*b_y)) + eps
    ux, uy, uz = b_x / den, float(SAD_mm) / den, b_y / den
    d1 = (rx*ux + ry*uy + rz*uz).astype(np.float32)
    d2 = np.sqrt(np.maximum(r2 - d1*d1, 0.0)).astype(np.float32)
    return d1, d2


def compute_d1_out_on_patient_grid(d1, d2, body_mask, ray_radius_mm=3.0):
    """
    d1_out = body_mask * t_entry (scalar).
    t_entry = minimum d1 value among body voxels near the beamlet ray axis.
    Matches the fix_sampling training convention: d1_out is uniform inside
    the body (= air path length before entering), zero outside.
    """
    body = body_mask.astype(bool)
    near_ray = d2 <= float(ray_radius_mm)
    inside_ray = body & near_ray
    if not np.any(inside_ray):
        return np.zeros_like(d1, dtype=np.float32)
    t_entry = float(np.min(d1[inside_ray]))
    d1_out = np.zeros_like(d1, dtype=np.float32)
    d1_out[body] = t_entry
    return d1_out