import SimpleITK as sitk
import numpy as np
import os
from skimage.transform import resize
import portpy.photon as pp
import argparse

def get_ct_image(ct: pp.CT):
    ct_arr = ct.ct_dict['ct_hu_3d'][0]
    ct_image = sitk.GetImageFromArray(ct_arr)
    ct_image.SetOrigin(ct.ct_dict['origin_xyz_mm'])
    ct_image.SetSpacing(ct.ct_dict['resolution_xyz_mm'])
    ct_image.SetDirection(ct.ct_dict['direction'])

    return ct_image


def resample(img, ref_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetReferenceImage(ref_image)
    img = resampler.Execute(img)

    return img


def resample_dose(dose, ref_dose):
    dims = ref_dose.shape
    dose = np.moveaxis(dose, 0, -1)  # Channels last
    expected_shape = (dims[1], dims[2], dims[0])
    dose = resize(dose, expected_shape, order=1, preserve_range=True, anti_aliasing=False)
    dose = np.moveaxis(dose, -1, 0)

    return dose


def crop_arr(img, start, end, is_mask=False):
    # Crop to setting given by start/end coordinates list, assuming depth,height,width
    img_arr = sitk.GetArrayFromImage(img)
    img_cropped = img_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]
    img_cropped = sitk.GetImageFromArray(img_cropped)
    img_cropped.SetOrigin(img.GetOrigin())
    img_cropped.SetDirection(img.GetDirection())
    img_cropped.SetSpacing(img.GetSpacing())
    # img_cropped = np.moveaxis(img_cropped, 0, -1)  # Slices last
    #
    # order = 0
    # if is_mask is False:
    #     order = 1
    # img_resized = resize(img_cropped, (128, 128, 128), order=order, preserve_range=True, anti_aliasing=False).astype(np.float32)
    # if is_mask is True:
    #     img_resized = img_resized.astype(np.uint8)
    #
    # img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

    return img_cropped

def crop_resize_img(img, start, end, is_mask=False, size_xyz = (128, 128, 128)):
    # Crop to setting given by start/end coordinates list, assuming depth,height,width

    img_cropped = img[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]
    img_cropped = np.moveaxis(img_cropped, 0, -1)  # Slices last

    order = 0
    if is_mask is False:
        order = 1
    img_resized = resize(img_cropped, size_xyz, order=order, preserve_range=True, anti_aliasing=False).astype(
        np.float32)
    if is_mask is True:
        if len(np.unique(np.sort(img_resized))) <= 2:
            img_resized = img_resized.astype(np.uint8)

    img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

    return img_resized


def crop_resize_img_with_mask(img, start, end, is_mask=False, img_with_mask=False, mask=np.ones((128, 128, 128))):
    # Crop to setting given by start/end coordinates list, assuming depth,height,width

    img_cropped = img[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]
    if img_with_mask:
        mask_cropped = mask[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]
        img_resized = downsample_3d_masked(img_cropped, (128, 128, 128), mask=mask_cropped)
    else:
        # original cose
        img_cropped = np.moveaxis(img_cropped, 0, -1)  # Slices last

        order = 0
        if is_mask is False:
            order = 1
        img_resized = resize(img_cropped, (128, 128, 128), order=order, preserve_range=True, anti_aliasing=False).astype(
            np.float32)
        if is_mask is True:
            if len(np.unique(np.sort(img_resized))) <= 2:
                img_resized = img_resized.astype(np.uint8)

        img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

    return img_resized

# new strategy downsampling. Fixed the old one.
def np_zyx_to_sitk(img_zyx: np.ndarray, origin_xyz, spacing_xyz, direction):
    img = sitk.GetImageFromArray(img_zyx.astype(np.float32))
    img.SetOrigin(tuple(origin_xyz))
    img.SetSpacing(tuple(spacing_xyz))
    img.SetDirection(tuple(direction))
    return img


def sitk_to_np_zyx(img_sitk):
    return sitk.GetArrayFromImage(img_sitk).astype(np.float32)


# def make_isocenter_reference_image(
#     iso_xyz_mm,
#     spacing_xyz_mm=(2.0, 2.0, 2.0),
#     size_xyz=(128, 128, 128),
#     direction=None,
# ):
#     """
#     Creates a fixed-size physical box centered at isocenter.
#     size_xyz is in (x, y, z), but SITK array output will still become Z,Y,X after GetArrayFromImage.
#     """
#     iso_xyz_mm = np.asarray(iso_xyz_mm, dtype=np.float64)
#     spacing_xyz_mm = np.asarray(spacing_xyz_mm, dtype=np.float64)
#     size_xyz = np.asarray(size_xyz, dtype=np.int32)
#
#     if direction is None:
#         direction = np.eye(3, dtype=np.float64).reshape(-1).tolist()
#
#     # place voxel centers symmetrically around isocenter
#     half_extent_xyz_mm = spacing_xyz_mm * (size_xyz - 1) / 2.0
#     origin_xyz_mm = iso_xyz_mm - half_extent_xyz_mm
#
#     ref = sitk.Image(
#         [int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])],
#         sitk.sitkFloat32
#     )
#     ref.SetOrigin(tuple(origin_xyz_mm))
#     ref.SetSpacing(tuple(spacing_xyz_mm))
#     ref.SetDirection(tuple(direction))
#
#     return ref, {
#         "origin_xyz_mm": tuple(origin_xyz_mm.tolist()),
#         "spacing_xyz_mm": tuple(spacing_xyz_mm.tolist()),
#         "size_xyz": tuple(size_xyz.tolist()),
#         "iso_xyz_mm": tuple(iso_xyz_mm.tolist()),
#     }


def make_reference_image_from_center(
    center_xyz_mm,
    spacing_xyz_mm=(2.0, 2.0, 2.0),
    size_xyz=(160, 160, 96),
    direction=None,
):
    """
    Create fixed-size reference image centered at center_xyz_mm.
    size_xyz is (X, Y, Z).
    """
    center_xyz_mm = np.asarray(center_xyz_mm, dtype=np.float64)
    spacing_xyz_mm = np.asarray(spacing_xyz_mm, dtype=np.float64)
    size_xyz = np.asarray(size_xyz, dtype=np.int32)

    if direction is None:
        direction = np.eye(3, dtype=np.float64).reshape(-1).tolist()

    half_extent_xyz_mm = spacing_xyz_mm * (size_xyz - 1) / 2.0
    origin_xyz_mm = center_xyz_mm - half_extent_xyz_mm

    ref = sitk.Image(
        [int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])],
        sitk.sitkFloat32
    )
    ref.SetOrigin(tuple(origin_xyz_mm))
    ref.SetSpacing(tuple(spacing_xyz_mm))
    ref.SetDirection(tuple(direction))

    return ref, {
        "origin_xyz_mm": tuple(origin_xyz_mm.tolist()),
        "spacing_xyz_mm": tuple(spacing_xyz_mm.tolist()),
        "size_xyz": tuple(size_xyz.tolist()),
        "center_xyz_mm": tuple(center_xyz_mm.tolist()),
    }

def get_body_center_xy_mm(structs, ct):
    """
    Return body center in x/y using BODY mask bounding box midpoint.
    z is not returned because we will use iso z.
    """
    body_mask = structs.get_structure_mask_3d('BODY')  # Z,Y,X
    z_idx, y_idx, x_idx = np.where(body_mask > 0)

    if len(x_idx) == 0:
        raise ValueError("BODY mask is empty.")

    spacing_xyz = np.asarray(ct.ct_dict['resolution_xyz_mm'], dtype=np.float64)
    origin_xyz = np.asarray(ct.ct_dict['origin_xyz_mm'], dtype=np.float64)

    # x_center_idx = 0.5 * (x_idx.min() + x_idx.max())
    # y_center_idx = 0.5 * (y_idx.min() + y_idx.max())

    x_center_idx = x_idx.mean() # it is more robust as body can include artifacts and can be irregularly shaped, so mean is better than midpoint of min/max
    y_center_idx = y_idx.mean()

    x_center_mm = origin_xyz[0] + x_center_idx * spacing_xyz[0]
    y_center_mm = origin_xyz[1] + y_center_idx * spacing_xyz[1]

    return float(x_center_mm), float(y_center_mm)

def resample_to_reference(img_sitk, ref_img, is_mask=False, default_value=0.0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    return resampler.Execute(img_sitk)


def resample_ct_to_fixed_iso_box(ct: pp.CT, ref_img, default_ct_value=-1000.0):
    ct_img = get_ct_image(ct)
    out = resample_to_reference(ct_img, ref_img, is_mask=False, default_value=default_ct_value)
    return sitk_to_np_zyx(out)


def resample_array_on_ct_grid_to_fixed_iso_box(
    arr_zyx: np.ndarray,
    ct: pp.CT,
    ref_img,
    is_mask=False,
    default_value=0.0,
):
    """
    arr_zyx is assumed aligned to the CT grid in the same Z,Y,X indexing as ct.ct_dict['ct_hu_3d'][0].
    """
    img_sitk = np_zyx_to_sitk(
        arr_zyx,
        origin_xyz=ct.ct_dict['origin_xyz_mm'],
        spacing_xyz=ct.ct_dict['resolution_xyz_mm'],
        direction=ct.ct_dict['direction'],
    )
    out = resample_to_reference(img_sitk, ref_img, is_mask=is_mask, default_value=default_value)
    out_np = sitk_to_np_zyx(out)

    if is_mask:
        out_np = (out_np > 0.5).astype(np.uint8)

    return out_np

def resample_masked_array_on_ct_grid_to_fixed_iso_box(
    arr_zyx: np.ndarray,
    valid_mask_zyx: np.ndarray,
    ct,
    ref_img,
    default_value=0.0,
    eps=1e-6,
    valid_threshold=1e-3,
):
    """
    Mask-aware linear interpolation:
      out = interp(arr * valid) / interp(valid)

    Inputs are assumed aligned to CT grid in Z,Y,X order.
    valid_mask_zyx should be 1 where arr is defined, 0 elsewhere.
    """

    origin_xyz = ct.ct_dict['origin_xyz_mm']
    spacing_xyz = ct.ct_dict['resolution_xyz_mm']
    direction = ct.ct_dict['direction']

    arr_zyx = arr_zyx.astype(np.float32)
    valid_mask_zyx = valid_mask_zyx.astype(np.float32)

    weighted_zyx = arr_zyx * valid_mask_zyx

    weighted_img = np_zyx_to_sitk(weighted_zyx, origin_xyz, spacing_xyz, direction)
    valid_img = np_zyx_to_sitk(valid_mask_zyx, origin_xyz, spacing_xyz, direction)

    weighted_res = resample_to_reference(
        weighted_img, ref_img, is_mask=False, default_value=0.0
    )
    valid_res = resample_to_reference(
        valid_img, ref_img, is_mask=False, default_value=0.0
    )

    weighted_np = sitk_to_np_zyx(weighted_res)
    valid_np = sitk_to_np_zyx(valid_res)

    out = np.full_like(weighted_np, default_value, dtype=np.float32)
    good = valid_np > valid_threshold
    out[good] = weighted_np[good] / (valid_np[good] + eps)

    return out, valid_np

####### new strategy for getting directly downsampled distances.
import numpy as np


def get_fixed_box_voxel_centers_xyz(ref_meta):
    """
    Returns voxel center coordinates for the fixed box in patient coordinates.

    ref_meta must contain:
      - origin_xyz_mm
      - spacing_xyz_mm
      - size_xyz   (X, Y, Z)

    Returns:
      Xg, Yg, Zg as arrays in ZYX shape
    """
    origin = np.asarray(ref_meta["origin_xyz_mm"], dtype=np.float32)
    spacing = np.asarray(ref_meta["spacing_xyz_mm"], dtype=np.float32)
    size_xyz = np.asarray(ref_meta["size_xyz"], dtype=np.int32)

    nx, ny, nz = int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])

    x = origin[0] + spacing[0] * np.arange(nx, dtype=np.float32)
    y = origin[1] + spacing[1] * np.arange(ny, dtype=np.float32)
    z = origin[2] + spacing[2] * np.arange(nz, dtype=np.float32)

    # Make arrays in Z,Y,X order to match numpy volume convention
    Zg, Yg, Xg = np.meshgrid(z, y, x, indexing='ij')

    return Xg, Yg, Zg


def get_beam_frame_coords_from_fixed_box(Xg, Yg, Zg, iso_xyz_mm, gantry, couch=0, collimator=None):
    """
    Convert fixed-box voxel centers from patient coords to beam coords.

    Inputs:
      Xg, Yg, Zg: ZYX-shaped arrays of patient coordinates
      iso_xyz_mm: (3,) array
    Returns:
      xp, yp, zp in ZYX shape
    """

    iso_xyz_mm = np.asarray(iso_xyz_mm, dtype=np.float32)
    R = get_rotation_matrix(gantry, couch, collimator).astype(np.float32)

    pts = np.stack([
        Xg.ravel() - iso_xyz_mm[0],
        Yg.ravel() - iso_xyz_mm[1],
        Zg.ravel() - iso_xyz_mm[2]
    ], axis=1)  # (N, 3)

    rb = pts @ R
    xp = rb[:, 0].reshape(Xg.shape)
    yp = rb[:, 1].reshape(Xg.shape)
    zp = rb[:, 2].reshape(Xg.shape)

    return xp, yp, zp


def compute_d1_d2_on_fixed_grid(
    xp, yp, zp,
    b_x, b_y,
    SAD_mm,
    eps=1e-12
):
    """
    Compute d1/d2 directly on fixed downsampled grid for one beamlet.

    xp, yp, zp are beam-frame voxel coordinates in ZYX shape.
    b_x, b_y are beamlet center coordinates in BEV/isocenter plane.
    """
    rx = xp
    ry = yp + SAD_mm
    rz = zp

    den = np.sqrt(b_x * b_x + SAD_mm * SAD_mm + b_y * b_y) + eps
    ux = b_x / den
    uy = SAD_mm / den
    uz = b_y / den

    d1 = rx * ux + ry * uy + rz * uz
    r2 = rx * rx + ry * ry + rz * rz
    d2 = np.sqrt(np.maximum(r2 - d1 * d1, 0.0))

    return d1.astype(np.float32), d2.astype(np.float32)


def precompute_beam_geometry_on_fixed_grid(ref_meta, iso_xyz_mm, gantry, couch=0, collimator=None, SAD_mm=1000.0):
    """
    Precompute beam-dependent geometry once per beam.
    """
    Xg, Yg, Zg = get_fixed_box_voxel_centers_xyz(ref_meta)
    xp, yp, zp = get_beam_frame_coords_from_fixed_box(
        Xg, Yg, Zg,
        iso_xyz_mm=iso_xyz_mm,
        gantry=gantry,
        couch=couch,
        collimator=collimator
    )

    rx = xp.astype(np.float32)
    ry = (yp + SAD_mm).astype(np.float32)
    rz = zp.astype(np.float32)
    r2 = (rx * rx + ry * ry + rz * rz).astype(np.float32)

    return {
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "r2": r2,
    }


def compute_d1_d2_from_precomputed_geom(precomp_geom, b_x, b_y, SAD_mm, eps=1e-12):
    rx = precomp_geom["rx"]
    ry = precomp_geom["ry"]
    rz = precomp_geom["rz"]
    r2 = precomp_geom["r2"]

    den = np.sqrt(b_x * b_x + SAD_mm * SAD_mm + b_y * b_y) + eps
    ux = b_x / den
    uy = SAD_mm / den
    uz = b_y / den

    d1 = rx * ux + ry * uy + rz * uz
    d2 = np.sqrt(np.maximum(r2 - d1 * d1, 0.0))

    return d1.astype(np.float32), d2.astype(np.float32)

# using ray tracing to find body entry/exit for each ray, which is more accurate than using the fixed box as a proxy for body contour. This is used for computing d1/d2 for voxels outside the fixed box but still inside the body.
def point_xyz_to_zyx_idx(point_xyz, origin_xyz, spacing_xyz, shape_zyx):
    """
    Convert patient xyz point (mm) to nearest voxel index in ZYX.
    """
    x = int(np.round((point_xyz[0] - origin_xyz[0]) / spacing_xyz[0]))
    y = int(np.round((point_xyz[1] - origin_xyz[1]) / spacing_xyz[1]))
    z = int(np.round((point_xyz[2] - origin_xyz[2]) / spacing_xyz[2]))

    if z < 0 or z >= shape_zyx[0] or y < 0 or y >= shape_zyx[1] or x < 0 or x >= shape_zyx[2]:
        return None
    return (z, y, x)

def compute_ray_body_entry_exit(
    source_xyz_mm,
    u_xyz,
    body_mask_zyx,
    origin_xyz_mm,
    spacing_xyz_mm,
    step_mm=1.0,
    max_t_mm=2000.0,
):
    """
    March along r(t) = s + t u and find first/last intersection with body mask.
    body_mask_zyx is on the same grid defined by origin/spacing.
    Returns (t_entry, t_exit), or (None, None) if no hit.
    """
    source_xyz_mm = np.asarray(source_xyz_mm, dtype=np.float64)
    u_xyz = np.asarray(u_xyz, dtype=np.float64)
    u_xyz = u_xyz / (np.linalg.norm(u_xyz) + 1e-12)

    shape_zyx = body_mask_zyx.shape
    ts = np.arange(0.0, max_t_mm + step_mm, step_mm, dtype=np.float64)

    inside = np.zeros(ts.shape, dtype=bool)

    for i, t in enumerate(ts):
        p = source_xyz_mm + t * u_xyz
        idx = point_xyz_to_zyx_idx(
            p, origin_xyz_mm, spacing_xyz_mm, shape_zyx
        )
        if idx is None:
            continue
        inside[i] = bool(body_mask_zyx[idx])

    if not np.any(inside):
        return None, None

    first = np.argmax(inside)
    last = len(inside) - 1 - np.argmax(inside[::-1])

    t_entry = float(ts[first])
    t_exit = float(ts[last])

    return t_entry, t_exit

def split_d1_using_entry_exit(d1_3d_down, body_mask_down, t_entry, t_exit):
    """
    Exact split once t_entry/t_exit are known.
    """
    d1_out = np.minimum(d1_3d_down, t_entry).astype(np.float32)

    if t_exit is None:
        d1_in = np.zeros_like(d1_3d_down, dtype=np.float32)
    else:
        chord = max(t_exit - t_entry, 0.0)
        d1_in = np.clip(d1_3d_down - t_entry, 0.0, chord).astype(np.float32)

    d1_in[~body_mask_down.astype(bool)] = 0.0
    return d1_out, d1_in

def build_fast_dose_1d_to_3d_mapper(inf_matrix):
    dose_vox_map = inf_matrix.opt_voxels_dict['ct_to_dose_voxel_map'][0]
    valid_ct = dose_vox_map >= 0

    flat_valid_ct_idx = np.flatnonzero(valid_ct.ravel())
    flat_dose_idx = dose_vox_map.ravel()[flat_valid_ct_idx].astype(np.int64)
    ct_shape = dose_vox_map.shape

    return {
        "flat_valid_ct_idx": flat_valid_ct_idx,
        "flat_dose_idx": flat_dose_idx,
        "ct_shape": ct_shape,
    }


def fast_dose_1d_to_3d(dose_1d, mapper, dtype=np.float32):
    out = np.zeros(np.prod(mapper["ct_shape"]), dtype=dtype)
    out[mapper["flat_valid_ct_idx"]] = dose_1d[mapper["flat_dose_idx"]]
    return out.reshape(mapper["ct_shape"])

def split_d1_in_out(d1_3d_down, d2_3d_down, body_mask_down, ray_radius_mm=2.5):
    """
    Minimal approximation:
    - find first body voxel near the ray
    - use its d1 as t_entry
    """
    body_mask_down = body_mask_down.astype(bool)
    near_ray = d2_3d_down <= ray_radius_mm
    inside_ray = body_mask_down & near_ray

    if not np.any(inside_ray):
        # no body intersection found
        d1_out = d1_3d_down.astype(np.float32).copy()
        d1_in = np.zeros_like(d1_3d_down, dtype=np.float32)
        t_entry = None
        return d1_out, d1_in, t_entry

    t_entry = float(np.min(d1_3d_down[inside_ray]))

    # d1_out = np.minimum(d1_3d_down, t_entry).astype(np.float32)
    # d1_out only inside body voxels
    d1_out = np.zeros_like(d1_3d_down, dtype=np.float32)
    d1_out[body_mask_down] = t_entry

    d1_in = np.maximum(d1_3d_down - t_entry, 0.0).astype(np.float32)

    # keep inside-depth only inside body
    d1_in[~body_mask_down] = 0.0

    return d1_out, d1_in, t_entry

######################################################################

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom, minimum_filter
import matplotlib.pyplot as plt


def downsample_3d_masked(data_array, target_shape=(128, 128, 128), mask=np.ones((128, 128, 128))):
    """
    Downsamples a 3D NumPy array to a target shape, linearly interpolating
    only within the non-zero region (mask) while preserving all original
    values by clamping.

    Args:
        data_array (np.ndarray): The high-resolution 3D array.
        target_shape (tuple): The desired output shape (e.g., (128, 128, 128)).

    Returns:
        np.ndarray: The downsampled 3D array with value preservation.
    """
    original_shape = data_array.shape
    # body_mask = data_array > 0
    mask = mask.astype(bool)
    # Correctly calculate the scale factors for downsampling
    scale_factor = [t / o for t, o in zip(target_shape, original_shape)]

    # 1. Perform initial linear interpolation
    x_orig = np.arange(original_shape[0])
    y_orig = np.arange(original_shape[1])
    z_orig = np.arange(original_shape[2])

    interpolator = RegularGridInterpolator(
        (x_orig, y_orig, z_orig),
        data_array,
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    new_x = np.linspace(0, original_shape[0] - 1, target_shape[0])
    new_y = np.linspace(0, original_shape[1] - 1, target_shape[1])
    new_z = np.linspace(0, original_shape[2] - 1, target_shape[2])

    new_grid_x, new_grid_y, new_grid_z = np.meshgrid(new_x, new_y, new_z, indexing='ij')
    new_grid_coords = np.vstack((new_grid_x.ravel(), new_grid_y.ravel(), new_grid_z.ravel())).T

    # 2. Get the new body mask
    new_mask = zoom(mask.astype(float), scale_factor, order=0) > 0.5
    new_mask = new_mask.astype(bool)

    # Check for shape mismatch before indexing
    if new_grid_coords.shape[0] != new_mask.size:
        raise ValueError(
            f"Shape mismatch: new_grid_coords has shape {new_grid_coords.shape[0]} "
            f"but new_body_mask has shape {new_mask.size}"
        )

    # 3. Interpolate values
    masked_new_grid_coords = new_grid_coords[new_mask.ravel()]
    interpolated_values = interpolator(masked_new_grid_coords)

    # 4. Construct the base downsampled array
    resized_array = np.zeros(target_shape, dtype=data_array.dtype)
    resized_array[new_mask] = interpolated_values

    # --- Start of fix to preserve all values ---
    # The downsampling scale is how many original voxels map to one new voxel.
    filter_size = tuple(int(np.ceil(1 / sf)) for sf in scale_factor)

    # Apply a minimum filter to the original data, preserving the mask
    filtered_data = data_array.copy()
    filtered_data[~mask] = np.inf  # Use inf outside body to not influence min filter

    min_values = minimum_filter(filtered_data, size=filter_size)

    # Resample the min_values grid to the new shape using nearest-neighbor
    # The minimum value in the original neighborhood is placed into the new grid
    resampled_min_values = zoom(min_values.astype(float), scale_factor, order=0, prefilter=False)
    resampled_min_values = resampled_min_values.astype(data_array.dtype)
    resampled_min_values[~new_mask] = 0

    # 5. Clamping the resampled array to preserve minimums
    resized_array = np.maximum(resized_array, resampled_min_values)
    # --- End of fix ---

    return resized_array


def get_crop_settings_calc_box(ct: pp.CT, meta_data):
    cal_box_xyz_start = meta_data['opt_voxels']['cal_box_xyz_start']
    cal_box_xyz_end = meta_data['opt_voxels']['cal_box_xyz_end']
    ct_img = get_ct_image(ct)
    start_xyz = ct_img.TransformPhysicalPointToIndex(cal_box_xyz_start)  # X,Y,Z
    end_xyz = ct_img.TransformPhysicalPointToIndex(cal_box_xyz_end)  # X,Y,Z
    start_zyx = [start_xyz[2], start_xyz[1], start_xyz[0]]
    end_zyx = [end_xyz[2], end_xyz[1], end_xyz[0]]
    return start_zyx, end_zyx


def get_crop_settings(oar):
    # Use to get crop settings
    # Don't use cord or eso as they spread through more slices
    # If total number of slices is less than 128 then don't crop at all
    # Use start and end index from presence of any anatomy or ptv
    # If that totals more than 128 slices then leave as is.
    # If that totals less than 128 slices then add slices before and after to make total slices to 128

    oar1 = oar.copy()
    oar1[np.where(oar == 1)] = 0
    oar1[np.where(oar == 2)] = 0

    # For 2D cropping just do center cropping 256x256
    center = [0, oar.shape[1] // 2, oar1.shape[2] // 2]
    start = [0, center[1] - 150, center[2] - 150]
    end = [0, center[1] + 150, center[2] + 150]

    depth = oar1.shape[0]
    if depth < 128:
        start[0] = 0
        end[0] = depth

        return start, end

    first_slice = -1
    last_slice = -1
    for i in range(depth):
        frame = oar1[i]
        if np.any(frame):
            first_slice = i
            break
    for i in range(depth - 1, -1, -1):
        frame = oar1[i]
        if np.any(frame):
            last_slice = i
            break

    expanse = last_slice - first_slice + 1
    if expanse >= 128:
        start[0] = first_slice
        end[0] = last_slice

        return start, end

    # print('Get\'s here')
    slices_needed = 128 - expanse
    end_slices = slices_needed // 2
    beg_slices = slices_needed - end_slices

    room_available = depth - expanse
    end_room_available = depth - last_slice - 1
    beg_room_available = first_slice

    leftover_beg = beg_room_available - beg_slices
    if leftover_beg < 0:
        end_slices += np.abs(leftover_beg)
        first_slice = 0
    else:
        first_slice = first_slice - beg_slices

    leftover_end = end_room_available - end_slices
    if leftover_end < 0:
        first_slice -= np.abs(leftover_end)
        last_slice = depth - 1
    else:
        last_slice = last_slice + end_slices

    if first_slice < 0:
        first_slice = 0

    start[0] = first_slice
    end[0] = last_slice

    return start, end


def attach_slices(pred_dose, ct_img, start, end):
    ct_arr = sitk.GetArrayFromImage(ct_img)
    ref_dose_arr = np.zeros_like(ct_arr, dtype=float)
    # ref_dose_copy = ref_dose
    ref_dose_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]] = pred_dose
    # dose = ref_dose_arr
    # ref_dose = sitk.GetImageFromArray(ref_dose)
    dose = sitk.GetImageFromArray(ref_dose_arr)
    dose.SetOrigin(ct_img.GetOrigin())
    dose.SetDirection(ct_img.GetDirection())
    dose.SetSpacing(ct_img.GetSpacing())

    return dose

def get_rotation_matrix(gantry_deg: float, couch_deg: float, collimator_deg: float = None, clockwise: bool = True) -> np.ndarray:
    """
    Patient→beam rotation (LPS). Use with row vectors: rb = r @ R.
    Angles given as CLOCKWISE-positive if clockwise=True (default).
    Beam axis after rotation is the y' axis.
    Quick rule of thumb

    Column + active + CCW input → use R(+θ)

    Row + passive + CW input (your case) → still use the standard CCW R(+θ) on the right

    If you switch any one of those, you may need a transpose or −θ
    """
    # s = -1.0 if clockwise else 1.0  # flip to CCW-internal
    g = np.deg2rad(gantry_deg)  # gantry about patient z
    c = np.deg2rad(couch_deg)  # couch  about patient y

    Rg = np.array([[np.cos(g), -np.sin(g), 0.0],
                   [np.sin(g), np.cos(g), 0.0],
                   [0.0, 0.0, 1.0]], float)

    Rc = np.array([[np.cos(c), 0.0, np.sin(c)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(c), 0.0, np.cos(c)]], float)

    rot_mat = Rc @ Rg  # row-vector passive rotation
    # rot_mat = (Rc @ Rg).T
    if collimator_deg is not None:
        t = np.deg2rad(collimator_deg)
        r_coll = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]], float)
        rot_mat = rot_mat @ r_coll
        # rot_mat = rot_mat @ r_coll.T  # append the transpose on the right

    return rot_mat

# def rotate_bev_by_collimator(u: np.ndarray, v: np.ndarray,
#                              collimator_deg: float, clockwise: bool = True):
#     """
#     Rotate BEV coords (u,v) around the beam axis by the collimator angle.
#     Angles are CLOCKWISE-positive if clockwise=True.
#     """
#     t = np.deg2rad(-collimator_deg if clockwise else collimator_deg)  # CW→-θ for standard CCW rotation
#     uc = u*np.cos(t) - v*np.sin(t)
#     vc = u*np.sin(t) + v*np.cos(t)
#     return uc, vc

def get_bev_coords(vox_xyz_mm: np.ndarray, iso_xyz_mm: np.ndarray,
               gantry: float, couch: float, collimator: float = None,
               SAD_mm: float = 1000.0, project: bool = True):
    """

    :param vox_xyz_mm:
    :param iso_xyz_mm:
    :param gantry:
    :param couch:
    :param collimator:
    :param SAD_mm:
    :param project: if True, it will project x, z coordinates in iso center plane/BEV plane
    :return:
    """
    R = get_rotation_matrix(gantry, couch, collimator)      # include collimator about beam z'
    r = vox_xyz_mm - iso_xyz_mm                                  # translate to iso
    rb = r @ R                                                 # rotate to beam frame
    xp, yp, zp = rb[:, 0], rb[:, 1], rb[:, 2]  # beam frame: Y is beam axis
    if project:
        s = SAD_mm / (SAD_mm + yp)
        u = xp * s  # BEV horizontal
        v = zp * s  # BEV vertical
    else:
        u, v = xp, zp
    w = yp  # depth along beam axis
    return u, v, w

def resample_to_grid(img_sitk, origin, spacing, direction_3x3, size_xyz,
                     default_value=0.0, interp=sitk.sitkLinear):
    Nx, Ny, Nz = map(int, size_xyz)  # x,y,z

    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(img_sitk)  # keeps world coordinate conventions consistent
    res.SetOutputOrigin(tuple(origin))
    res.SetOutputSpacing(tuple(spacing))
    res.SetOutputDirection(direction_3x3.flatten().tolist())
    res.SetSize([Nx, Ny, Nz])
    res.SetInterpolator(interp)
    res.SetDefaultPixelValue(float(default_value))
    res.SetOutputPixelType(sitk.sitkFloat32)
    return res.Execute(img_sitk)

def build_ray_grid(source_lps, target_lps,
                   view_size_mm=(400.0, 80.0, 80.0),   # (Depth, Width, Height) in mm
                   out_size=(128, 64, 64),             # (Nx, Ny, Nz)  (x=depth)
                   x0_mm=0.0):                         # start distance from source along ray
    source_lps = np.asarray(source_lps, float)
    target_lps = np.asarray(target_lps, float)

    v = target_lps - source_lps
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("source and target too close.")
    ux = v / n  # ray direction -> output X axis

    # pick a stable arbitrary vector not parallel to ux
    arb = np.array([0.0, 0.0, 1.0]) if abs(ux[2]) < 0.9 else np.array([0.0, 1.0, 0.0])

    # right-handed basis: (ux, uy, uz)
    uy = np.cross(arb, ux)
    uy /= (np.linalg.norm(uy) + 1e-12)
    uz = np.cross(ux, uy)
    uz /= (np.linalg.norm(uz) + 1e-12)

    depth_mm, width_mm, height_mm = map(float, view_size_mm)
    Nx, Ny, Nz = map(int, out_size)

    # Direction matrix: columns are physical directions of output axes (x,y,z)
    D = np.stack([ux, uy, uz], axis=1)  # 3x3

    # Origin is the corner of the box so that the ray passes through the center of y/z
    origin = (source_lps
              + ux * float(x0_mm)
              - uy * (width_mm / 2.0)
              - uz * (height_mm / 2.0))

    # spacing: choose a consistent convention
    spacing = (depth_mm / Nx, width_mm / Ny, height_mm / Nz)

    return origin, spacing, D

import SimpleITK as sitk
import numpy as np

def estimate_body_entry_distance_mm_fast(body_mask_sitk: sitk.Image,
                                        source_lps: np.ndarray,
                                        beamlet_center_lps: np.ndarray,
                                        max_dist_mm: float = 2000.0,
                                        step_mm: float = 2.0) -> float | None:
    """
    Fast, vectorized BODY entry distance along the ray using numpy.
    Returns s (mm) from source where ray first hits BODY. None if no hit.
    """

    src = np.asarray(source_lps, float)
    tgt = np.asarray(beamlet_center_lps, float)
    v = tgt - src
    den = float(np.linalg.norm(v))
    if den < 1e-6:
        return None
    u = v / den

    # mask array and geometry
    mask_zyx = sitk.GetArrayFromImage(body_mask_sitk)  # (z,y,x)
    size_xyz = np.array(body_mask_sitk.GetSize(), dtype=int)  # (x,y,z)
    origin = np.array(body_mask_sitk.GetOrigin(), dtype=float)  # (x,y,z) in LPS mm
    spacing = np.array(body_mask_sitk.GetSpacing(), dtype=float)  # (x,y,z) mm
    D = np.array(body_mask_sitk.GetDirection(), dtype=float).reshape(3, 3)  # 3x3

    # sample distances along ray (stop a bit past the beamlet center)
    s_max = min(max_dist_mm, den + 1000.0)
    s_vals = np.arange(0.0, s_max + 1e-6, step_mm, dtype=float)  # (N,)

    # physical points along ray: p(s) = src + s*u
    pts = src[None, :] + s_vals[:, None] * u[None, :]  # (N,3)

    # physical -> continuous index in SITK index space (x,y,z)
    # (p - origin) @ D  gives (idx * spacing) in row-vector form
    idx_f = ((pts - origin[None, :]) @ D) / spacing[None, :]  # (N,3)

    # nearest-neighbor index
    idx = np.rint(idx_f).astype(int)  # (N,3) in (x,y,z)

    # bounds check
    inb = (
        (idx[:, 0] >= 0) & (idx[:, 0] < size_xyz[0]) &
        (idx[:, 1] >= 0) & (idx[:, 1] < size_xyz[1]) &
        (idx[:, 2] >= 0) & (idx[:, 2] < size_xyz[2])
    )
    if not np.any(inb):
        return None

    idx_in = idx[inb]                 # (M,3)
    s_in = s_vals[inb]                # (M,)

    # mask lookup: numpy is (z,y,x) => (iz,iy,ix)
    hit = mask_zyx[idx_in[:, 2], idx_in[:, 1], idx_in[:, 0]] > 0
    if not np.any(hit):
        return None

    first = int(np.flatnonzero(hit)[0])
    return float(s_in[first])


def patch_pred_to_ct_grid(pred_patch_zyx: np.ndarray,
                          origin_xyz_mm, spacing_xyz_mm, direction_3x3,
                          ct_ref_sitk: sitk.Image,
                          default_value=0.0, interp=sitk.sitkLinear) -> np.ndarray:
    """
    pred_patch_zyx: numpy array in (z,y,x) — same order as sitk.GetArrayFromImage gave you.
    origin/spacing/direction: the ray-patch grid params you used for cropping.
    ct_ref_sitk: original CT image (full resolution) used as reference grid.
    Returns: full dose in CT grid as numpy (z,y,x) float32
    """

    # 1) Make a SITK image in the PATCH physical space
    pred_sitk = sitk.GetImageFromArray(pred_patch_zyx.astype(np.float32))  # assumes array is z,y,x
    pred_sitk.SetOrigin(tuple(map(float, origin_xyz_mm)))
    pred_sitk.SetSpacing(tuple(map(float, spacing_xyz_mm)))
    pred_sitk.SetDirection(np.asarray(direction_3x3, float).reshape(3, 3).flatten().tolist())

    # 2) Resample PATCH -> CT grid (physical alignment does the work)
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(ct_ref_sitk)            # output grid = CT grid
    res.SetInterpolator(interp)
    res.SetDefaultPixelValue(float(default_value))
    res.SetTransform(sitk.Transform())            # identity
    full_pred_sitk = res.Execute(pred_sitk)

    # 3) Back to numpy on CT grid (z,y,x)
    full_pred_zyx = sitk.GetArrayFromImage(full_pred_sitk).astype(np.float32)
    return full_pred_zyx


# NOTE: the BEV / ray-aligned patch geometry lives in
# portpy.ai.inference.ray_geometry (build_ray_grid_with_beam_axes,
# compute_ray_patch_d1_d2_split, source_lps_from_beam, beamlet_center_lps,
# estimate_body_entry_distance_mm, resample_to_grid). The BEV preprocess imports
# it from there so training data is built by the SAME code the RayUNetPredictor
# uses at inference time -- do not re-implement it here.


# import torch
# import torch.nn.functional as F
# import numpy as np
#
#
# def compute_rpl_high_res(ct_array_np, ray_vector_np, voxel_spacing_mm):
#     """
#     Computes Radiological Path Length on the full resolution array.
#
#     :param ct_array_np: High-res numpy array of CT (HU values).
#     :param ray_vector_np: (3,) numpy array direction vector [dx, dy, dz].
#     :param voxel_spacing_mm: float or tuple, average step size in mm.
#     :return: High-res numpy array of RPL.
#     """
#
#     # 1. Setup Device (Use GPU if available, otherwise CPU)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 2. Convert HU to Relative Electron Density
#     #    Formula: (HU + 1000) / 1000, clamped at 0.
#     ct_tensor = torch.from_numpy(ct_array_np).float().to(device)
#     density_tensor = (ct_tensor + 1000.0) / 1000.0
#     density_tensor = torch.clamp(density_tensor, min=0.0)
#
#     D, H, W = density_tensor.shape
#
#     # 3. Prepare Ray Vector
#     ray_vector = torch.tensor(ray_vector_np, dtype=torch.float32, device=device)
#     ray_vector = ray_vector / (torch.norm(ray_vector) + 1e-8)
#
#     # 4. Rotation Logic: Align ray with Z-axis [0, 0, 1]
#     target_axis = torch.tensor([0.0, 0.0, 1.0], device=device)
#
#     # Check for parallel rays
#     if torch.allclose(ray_vector, target_axis, atol=1e-3):
#         rpl = torch.cumsum(density_tensor, dim=0) * voxel_spacing_mm
#         return rpl.cpu().numpy()
#
#     # Rotation Matrix (Rodrigues' Formula)
#     v = torch.linalg.cross(ray_vector, target_axis)
#     c = torch.dot(ray_vector, target_axis)
#     k = 1.0 / (1.0 + c + 1e-8)
#     vx, vy, vz = v[0], v[1], v[2]
#
#     R = torch.tensor([
#         [c + vx * vx * k, vx * vy * k - vz, vx * vz * k + vy],
#         [vy * vx * k + vz, c + vy * vy * k, vy * vz * k - vx],
#         [vz * vx * k - vy, vz * vy * k + vx, c + vz * vz * k]
#     ], device=device)
#
#     # 5. Rotate Volume -> Integrate -> Rotate Back
#     # We use affine_grid/grid_sample.
#     # Note: shape for grid is (N, C, D, H, W) -> (1, 1, D, H, W)
#
#     # Create Transform Matrix for Grid
#     theta = torch.eye(4, device=device)
#     theta[:3, :3] = R
#     theta = theta[:3, :].unsqueeze(0)  # (1, 3, 4)
#
#     # Grid Sample (Rotate to align ray with Z)
#     grid = F.affine_grid(theta, size=(1, 1, D, H, W), align_corners=True)
#
#     # Prepare input (N, C, D, H, W)
#     density_input = density_tensor.unsqueeze(0).unsqueeze(0)
#
#     # Sample
#     density_rotated = F.grid_sample(density_input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#
#     # Integrate (Cumsum) along Depth (dim 2)
#     # This creates the path length accumulation
#     rpl_rotated = torch.cumsum(density_rotated, dim=2) * voxel_spacing_mm[2]
#
#     # Rotate Back (Inverse Transform)
#     theta_inv = torch.eye(4, device=device)
#     theta_inv[:3, :3] = R.t()  # Transpose = Inverse for rotation matrix
#     theta_inv = theta_inv[:3, :].unsqueeze(0)
#
#     grid_inv = F.affine_grid(theta_inv, size=(1, 1, D, H, W), align_corners=True)
#     rpl_final = F.grid_sample(rpl_rotated, grid_inv, mode='bilinear', padding_mode='zeros', align_corners=True)
#
#     # 6. Return as Numpy
#     return rpl_final.squeeze().cpu().numpy()

import torch
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np


# # ====================================================================
# # CORRECTED RPL FUNCTION
# # Aligns beam ray with Z-axis (Dimension 0) for integration
# # ====================================================================
# def compute_rpl_high_res(
#         ct_array_np,
#         beam_direction_vector_np,  # [dx, dy, dz]
#         voxel_spacing_mm,  # (sx, sy, sz) where sx, sy, sz are X, Y, Z spacings
#         body_mask_np=None,
#         use_reverse_cumsum=True  # Standard accumulation: Source (Entrance) to Voxel
# ):
#     """
#     Computes Radiological Path Length (RPL/WED) using Volumetric Rotation.
#
#     Assumes CT array shape is (D, H, W) corresponding to (Z, Y, X) axes.
#     Target Axis (zhat) is [0, 0, 1] which aligns with Dimension 2 (W/X).
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # --- HU → relative density ---
#     # ... (HU conversion and clamping) ...
#     ct = torch.from_numpy(ct_array_np).float().to(device)
#     ct = torch.clamp(ct, min=-1000.0, max=3071.0)
#     rho = (ct + 1000.0) / 1000.0
#     rho = torch.clamp(rho, min=0.0, max=3.0)
#
#     # --- Voxel Spacing Unpack ---
#     sx_in, sy_in, sz_in = map(float, voxel_spacing_mm)
#     dz, dy, dx = sz_in, sy_in, sx_in  # Spacings for D (Z), H (Y), W (X)
#
#     # --- apply body mask ---
#     if body_mask_np is not None:
#         m = torch.from_numpy(body_mask_np).to(device).float()
#         m = (m > 0).float()
#         rho = rho * m
#
#     D, H, W = rho.shape
#
#     # --- beam dir to tensor ---
#     ray = torch.tensor(beam_direction_vector_np, dtype=torch.float32, device=device)
#     ray = ray / (torch.norm(ray) + 1e-8)
#
#     # 1. FIX: Target axis [0, 0, 1] (Aligns with Dimension 2 / W / X)
#     zhat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
#
#     if torch.allclose(ray, zhat, atol=1e-4):
#         # Aligned case: integrate along dim 2 (X) using dx spacing
#         rpl = torch.cumsum(rho, dim=2) * dx  # ⚠️ USE DX for X-accumulation
#         if use_reverse_cumsum:
#             rpl = torch.flip(torch.cumsum(torch.flip(rho, dims=[2]), dim=2), dims=[2]) * dx
#         if body_mask_np is not None:
#             rpl = rpl * m
#         return rpl.cpu().numpy()
#
#     # --- Rodrigues rotation ray → X-axis (dim 2) ---
#     v = torch.linalg.cross(ray, zhat)
#     c = torch.dot(ray, zhat)
#     s = torch.norm(v)
#     Vx = torch.tensor([[0.0, -v[2], v[1]],
#                        [v[2], 0.0, -v[0]],
#                        [-v[1], v[0], 0.0]], device=device)
#     R = torch.eye(3, device=device) + Vx + (Vx @ Vx) * ((1.0 - c) / (s ** 2 + 1e-8))
#
#     # --- rotate volume to align beam with X-axis (dim 2) ---
#     theta = torch.eye(4, device=device)
#     theta[:3, :3] = R
#     theta = theta[:3, :].unsqueeze(0)  # (1, 3, 4)
#
#     grid = F.affine_grid(theta, size=(1, 1, D, H, W), align_corners=True)
#
#     rho_in = rho.unsqueeze(0).unsqueeze(0)
#     rho_rot = F.grid_sample(rho_in, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
#
#     # --- integrate along depth (dim=2) using dx spacing ---
#     # ⚠️ Integration dimension is dim 2 (W) and use dx
#     if use_reverse_cumsum:
#         rpl_rot = torch.flip(torch.cumsum(torch.flip(rho_rot, dims=[2]), dim=2), dims=[2]) * dx
#     else:
#         rpl_rot = torch.cumsum(rho_rot, dim=2) * dx
#
#     # --- rotate back ---
#     theta_inv = torch.eye(4, device=device)
#     theta_inv[:3, :3] = R.t()
#     theta_inv = theta_inv[:3, :].unsqueeze(0)
#
#     grid_inv = F.affine_grid(theta_inv, size=(1, 1, D, H, W), align_corners=True)
#     rpl = F.grid_sample(rpl_rot, grid_inv, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
#
#     if body_mask_np is not None:
#         rpl = rpl * m
#
#     return rpl.detach().cpu().numpy()

def compute_rpl_high_res(
    ct_array_np,
    beam_direction_vector_np,          # [dx, dy, dz]
    voxel_spacing_mm,                  # (sx, sy, sz)
    body_mask_np=None,                 # optional bool/0-1 mask, same shape as CT
    use_reverse_cumsum=True            # accumulate from entrance to exit
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- HU → relative density (clip to scanner range; then to [0, 3]) ---
    ct = torch.from_numpy(ct_array_np).float().to(device)
    ct = torch.clamp(ct, min=-1000.0, max=3071.0)                    # your HU range
    rho = (ct + 1000.0) / 1000.0
    rho = torch.clamp(rho, min=0.0, max=3.0)

    # --- apply body mask (body-only distances) ---
    if body_mask_np is not None:
        m = torch.from_numpy(body_mask_np).to(device).float()
        # ensure 0/1:
        m = (m > 0).float()
        rho = rho * m

    D, H, W = rho.shape
    sx, sy, sz = map(float, voxel_spacing_mm)

    # --- beam dir to tensor ---
    ray = torch.tensor(beam_direction_vector_np, dtype=torch.float32, device=device)
    ray = ray / (torch.norm(ray) + 1e-8)

    zhat = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)  # align beam to -Z so slice 0 is entrance
    if torch.allclose(ray, zhat, atol=1e-4):
        rpl = torch.cumsum(rho, dim=0) * sz
        if use_reverse_cumsum:
            # flip so entrance→exit accumulation if your index 0 is exit side
            rpl = torch.flip(torch.cumsum(torch.flip(rho, dims=[0]), dim=0), dims=[0]) * sz
        # mask again to zero any interpolation bleed (if any)
        if body_mask_np is not None:
            rpl = rpl * m
        return rpl.cpu().numpy()

    # --- Rodrigues rotation ray → -Z ---
    v = torch.linalg.cross(ray, zhat)
    c = torch.dot(ray, zhat)
    s = torch.norm(v)
    Vx = torch.tensor([[0.0, -v[2],  v[1]],
                       [v[2],  0.0, -v[0]],
                       [-v[1], v[0], 0.0]], device=device)
    R = torch.eye(3, device=device) + Vx + (Vx @ Vx) * ((1.0 - c) / (s**2 + 1e-8))

    # --- rotate volume to align beam with -Z ---
    theta = torch.eye(3, 4, device=device)      # 3x4
    theta[:3, :3] = R
    grid = F.affine_grid(theta.unsqueeze(0), size=(1, 1, D, H, W), align_corners=True)

    rho_in = rho.unsqueeze(0).unsqueeze(0)      # (1,1,D,H,W)
    rho_rot = F.grid_sample(rho_in, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # --- integrate along depth (dim=2) ---
    if use_reverse_cumsum:
        rpl_rot = torch.flip(torch.cumsum(torch.flip(rho_rot, dims=[2]), dim=2), dims=[2]) * sz
    else:
        rpl_rot = torch.cumsum(rho_rot, dim=2) * sz

    # --- rotate back ---
    theta_inv = torch.eye(3, 4, device=device)
    theta_inv[:3, :3] = R.t()
    grid_inv = F.affine_grid(theta_inv.unsqueeze(0), size=(1, 1, D, H, W), align_corners=True)
    rpl = F.grid_sample(rpl_rot, grid_inv, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()

    # --- enforce body-only again to remove any interpolation bleed at edges ---
    if body_mask_np is not None:
        rpl = rpl * m

    return rpl.detach().cpu().numpy()


# # ----- signed distances to beamlet centers (d1=vertical y, d2=horizontal x) -----
# def d1_d2(u, v, bx_mm, by_mm):
#     d2 = u[:, None] - np.ravel(bx_mm)[None, :]  # horizontal to beamlet centers
#     d1 = v[:, None] - np.ravel(by_mm)[None, :]  # vertical to beamlet centers
#     return d1, d2

# ---------- OPTION A: uniform grid on BEV plane ----------
def get_beamlets_from_grid(structs: pp.Structures, vox_coords_mm=None, ct_hu_3d=None,    # optional if you want body-based bounds
                       iso_xyz_mm=None, gantry_deg=0.0, couch_deg=0.0, collimator_deg=None,
                       SAD_mm=1000.0, step_x_mm=2.5, step_y_mm=2.5,
                       jaw_rect=None, margin_mm=5.0, struct_name: str = 'BODY'):
    """
    Make a rectangular beamlet grid on the BEV (isocenter) plane.
    - If use_body_mask=True, ct_hu_3d & vox_coords_mm & get_bev_coords(...) must be provided to bound coverage.
    - Else, you must pass jaw_rect={'xmin','xmax','ymin','ymax'} (mm) on BEV plane.

    Returns: bx_mm, by_mm, width_mm, height_mm  (1D arrays)
    """
    if structs is not None:
        ind1 = structs.structures_dict['name'].index(struct_name)
        struct_vox_ind = structs.opt_voxels_dict['voxel_idx'][ind1]
    u, v, _ = get_bev_coords(vox_coords_mm, iso_xyz_mm,
                             gantry=gantry_deg, couch=couch_deg,
                             collimator=None, SAD_mm=SAD_mm, project=True)
    u, v = u[struct_vox_ind], v[struct_vox_ind]

    # --- 3) Bounds from structure projection (+ margin), optionally intersect with jaws ---
    xmin, xmax = u.min() - margin_mm, u.max() + margin_mm
    ymin, ymax = v.min() - margin_mm, v.max() + margin_mm

    if jaw_rect is not None:
        xmin = max(xmin, jaw_rect['xmin'])
        xmax = min(xmax, jaw_rect['xmax'])
        ymin = max(ymin, jaw_rect['ymin'])
        ymax = min(ymax, jaw_rect['ymax'])

    if not (xmax > xmin and ymax > ymin):
        return np.array([]), np.array([]), np.array([]), np.array([])

    # --- 4) Tile a uniform grid on BEV within those bounds ---
    xs = np.arange(xmin + 0.5 * step_x_mm, xmax - 0.5 * step_x_mm + 1e-9, step_x_mm)
    ys = np.arange(ymin + 0.5 * step_y_mm, ymax - 0.5 * step_y_mm + 1e-9, step_y_mm)
    if xs.size == 0 or ys.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    XX, YY = np.meshgrid(xs, ys)
    bx_mm = XX.ravel()
    by_mm = YY.ravel()
    width_mm = np.full_like(bx_mm, step_x_mm, dtype=float)
    height_mm = np.full_like(by_mm, step_y_mm, dtype=float)
    return bx_mm, by_mm, width_mm, height_mm


# ---------- OPTION B: leaf rows (beamlet height = leaf row height) ----------
def get_beamlets_from_leafs(
    leaf_y_mm,
    jaw_rect,
    step_x_mm=2.5,                # beamlet width across X (free)
    step_y_mm=2.5,                # MINIMUM beamlet height along Y; if None, use leaf row height
    scale_to_iso = 1,
    leaf_left_mm=None,            # NEW: per-row left leaf X at the same plane as leaf_y_mm
    leaf_right_mm=None            # NEW: per-row right leaf X at the same plane as leaf_y_mm
):
    """
    Build beamlets on the BEV (isocenter) plane using MLC leaf TIP Y-positions.

    - leaf_y_mm: 1D array of leaf *tip* Y positions (mm). If given at the MLC plane,
                 we project to isocenter by (SAD_mm / s_mlc).
    - jaw_rect: dict with {'xmin','xmax','ymin','ymax'} on BEV (mm).
    - step_x_mm: beamlet width (X).
    - step_y_mm: MINIMUM beamlet height (Y). Each row uses height = max(row_height_from_tips, step_y_mm).
                 If None, use the row height from adjacent tips.
    - eclipse_order: reverse row order to match Eclipse’s indexing (optional).
    - s_mlc: distance of source to MLC (for Y/X scaling to BEV).
    - leaf_left_mm / leaf_right_mm: OPTIONAL 1D arrays (length = number of rows = len(leaf_y_mm)-1)
      of per-row X leaf edges (same plane as leaf_y_mm). If provided, beamlets are created only
      inside [left, right] ∩ jaws for each row.

    Returns:
        dict with keys: 'id', 'bx_mm', 'by_mm', 'width_mm', 'height_mm'
    """
    import numpy as np

    # scale_to_iso = float(SAD_mm) / float(s_mlc)

    # --- tip positions → row heights & centers (project to BEV) ---
    y_tip = np.asarray(leaf_y_mm, dtype=float) * scale_to_iso
    y_tip = np.sort(y_tip)
    leaf_w = np.abs(np.diff(y_tip))  # per-leaf (tip-to-tip) heights
    y_edges = y_tip  # boundaries at tips

    if step_y_mm is None:
        # original behavior: one row per adjacent tip pair
        row_h = leaf_w.copy()
        row_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    else:
        # pack consecutive leaves until accumulated height >= step_y_mm (overshoot allowed)
        target = float(step_y_mm)
        rows_c, rows_h = [], []
        i = 0
        n = leaf_w.size
        while i < n:
            j = i
            acc = 0.0
            start = y_edges[i]
            while j < n and acc < target:
                acc += leaf_w[j]
                j += 1
            end = y_edges[j]  # j progressed at least one leaf (since step_y_mm > 0)
            rows_h.append(end - start)
            rows_c.append(0.5 * (start + end))
            i = j
        row_h = np.asarray(rows_h, dtype=float)
        row_c = np.asarray(rows_c, dtype=float)
    # =========================

    # --- optional per-row MLC X openings (project to BEV) ---
    use_mlc_openings = (leaf_left_mm is not None) and (leaf_right_mm is not None)
    if use_mlc_openings:
        L = np.asarray(leaf_left_mm, dtype=float) * scale_to_iso
        R = np.asarray(leaf_right_mm, dtype=float) * scale_to_iso
        if L.shape[0] != (row_c.shape[0]) or R.shape[0] != (row_c.shape[0]):
            raise ValueError("leaf_left_mm / leaf_right_mm must have length equal to the number of Y rows")
    else:
        L = R = None

    # # --- Eclipse order if requested ---
    # if eclipse_order:
    #     row_h = row_h[::-1]
    #     row_c = row_c[::-1]
    #     if use_mlc_openings:
    #         L = L[::-1]
    #         R = R[::-1]

    # --- clip rows to jaws using the effective height ---
    ymin, ymax = float(jaw_rect['ymin']), float(jaw_rect['ymax'])
    keep = (row_c + 0.5 * row_h > ymin - 1e-9) & (row_c - 0.5 * row_h < ymax + 1e-9)
    row_c, row_h = row_c[keep], row_h[keep]
    if use_mlc_openings:
        L, R = L[keep], R[keep]

    # --- tile along X for each kept row ---
    import math
    xmin, xmax = float(jaw_rect['xmin']), float(jaw_rect['xmax'])
    w = float(step_x_mm)

    # anchor grid to multiple of w at/left of xmin
    left0 = np.floor(xmin / w) * w
    # count so rightmost partial beamlet is included (like Ceil math in C#)
    xs_count = int(np.ceil(-xmin / w) + np.ceil(xmax / w))
    # global left edges and centers
    left_edges_global = left0 + w * np.arange(xs_count)
    centers_global = left_edges_global + 0.5 * w

    bx_list, by_list, w_list, h_list = [], [], [], []
    for i, (yc, hr) in enumerate(zip(row_c, row_h)):
        # allowed interval for this row:
        if use_mlc_openings:
            xl, xr = (L[i], R[i]) if L[i] <= R[i] else (R[i], L[i])
            x_left = max(xmin, xl)
            x_right = min(xmax, xr)
        else:
            x_left, x_right = xmin, xmax

        # keep any beamlet that OVERLAPS [x_left, x_right] (no clipping; same as C#)
        # i.e., (center + 0.5w > x_left) & (center - 0.5w < x_right)
        keep_x = (centers_global + 0.5 * w > x_left) & (centers_global - 0.5 * w < x_right)
        if not np.any(keep_x):
            continue

        centers = centers_global[keep_x]
        n = centers.size

        bx_list.append(centers)
        by_list.append(np.full(n, yc))
        w_list.append(np.full(n, w))  # fixed width w (no edge clipping; matches C#)
        h_list.append(np.full(n, hr))

    bx_mm     = np.concatenate(bx_list)
    by_mm     = np.concatenate(by_list)
    width_mm  = np.concatenate(w_list)
    height_mm = np.concatenate(h_list)

    beamlets = {
        'id': np.arange(len(bx_mm), dtype=int),
        'bx_mm': bx_mm,
        'by_mm': by_mm,
        'width_mm': width_mm,
        'height_mm': height_mm
    }
    return beamlets

# alpha = 0.1
# m = mask.bool()
# hot_true = (y_norm >= alpha) & m
# # ratio r = d2/d1 (robust to d1≈0)
# eps = 1e-6
# ratio = d2 / torch.clamp(d1, min=eps)
# # keep only masked voxels; drop inf/nan for plotting
# r_all  = ratio[m]
# r_all  = r_all[torch.isfinite(r_all)]
# r_hot  = ratio[hot_true]
# r_hot  = r_hot[torch.isfinite(r_hot)]
# r_cold = ratio[m & (~hot_true)]
# r_cold = r_cold[torch.isfinite(r_cold)]
# # (optional) clip tails for readable axes
# hi = torch.quantile(r_all, 0.995).item()
# r_all_np  = torch.clamp(r_all, 0, hi).cpu().numpy()
# r_hot_np  = torch.clamp(r_hot, 0, hi).cpu().numpy()
# r_cold_np = torch.clamp(r_cold, 0, hi).cpu().numpy()
# # threshold candidates
# # 1) match prevalence of hot voxels
# p_star = hot_true.float().mean().item() if hot_true.numel() else 0.2
# tau_q  = torch.quantile(r_all, p_star).item()
# # 2) simple F1-driven threshold sweep over percentiles (fast & optional)
# qs = torch.linspace(0.01, 0.99, 99)
# taus = torch.quantile(r_all, qs)
# best_f1, tau_f1 = -1.0, tau_q
# for t in taus:
#     hp = (ratio <= t) & m
#     tp = (hp & hot_true).sum().item()
#     fp = (hp & (~hot_true)).sum().item()
#     fn = ((~hp) & hot_true).sum().item()
#     f1 = (2*tp) / (2*tp + fp + fn + 1e-9)
#     if f1 > best_f1:
#         best_f1, tau_f1 = f1, float(t.item())
# print(f"[ratio d2/d1] tau_q (prevalence-match) = {tau_q:.4f}")
# print(f"[ratio d2/d1] tau_f1 (best F1 vs GT)    = {tau_f1:.4f}  (F1={best_f1:.3f})")

# # ---- histogram ----
# bins = 60
# plt.figure(figsize=(7,4))
# plt.hist(r_all_np, bins=bins, alpha=0.35, color='gray', label='all (masked)', density=True)
# if r_hot_np.size:
#     plt.hist(r_hot_np, bins=bins, alpha=0.45, label='hot (GT)', density=True)
# if r_cold_np.size:
#     plt.hist(r_cold_np, bins=bins, alpha=0.45, label='scatter (GT)', density=True)
# plt.axvline(tau_q,  color='k',  ls='--', lw=1.2, label=f'tau_q={tau_q:.3f}')
# plt.axvline(tau_f1, color='r',  ls='-.', lw=1.2, label=f'tau_F1={tau_f1:.3f}')
# plt.xlabel('ratio = d2 / d1'); plt.ylabel('density'); plt.title('Histogram of d2/d1 inside mask')
# plt.legend(); plt.tight_layout(); plt.show()


# to get overall mask tau
# eps = 1e-6
# ratio = d2 / torch.clamp(d1, min=eps)   # use raw d1,d2 (not per-channel normalized)
# mask_true = mask.bool()                 # your dose-derived overall mask
# # keep finite values and align both tensors
# valid = torch.isfinite(ratio)
# r_use = ratio[valid]
# y_use = mask_true[valid]
# # --- 1) prevalence-matched threshold (baseline) ---
# p_star = y_use.float().mean().item() if y_use.numel() else 0.2
# tau_q  = torch.quantile(r_use, p_star).item()
# # --- 2) best-F1 threshold (this is where we compare to TRUE mask) ---
# qs   = torch.linspace(0.01, 0.99, 99)
# taus = torch.quantile(r_use, qs)
# best_f1, tau_f1 = -1.0, tau_q
# for t in taus:
#     pred = (r_use <= t)                    # geometry-only predicted mask
#     tp = (pred & y_use).sum().item()       # ← compare to TRUE mask
#     fp = (pred & (~y_use)).sum().item()    # ← compare to TRUE mask
#     fn = ((~pred) & y_use).sum().item()    # ← compare to TRUE mask
#     f1 = (2*tp) / (2*tp + fp + fn + 1e-9)
#     if f1 > best_f1:
#         best_f1, tau_f1 = f1, float(t.item())
# tau_star = tau_f1
# print(f"[overall mask] tau_q={tau_q:.4f}  tau_F1={tau_star:.4f} (F1={best_f1:.3f})")
# # --- histogram for visualization ---
# hi = r_use.max()
# r_all  = torch.clamp(r_use, 0, hi).cpu().numpy()
# r_pos  = torch.clamp(r_use[y_use], 0, hi).cpu().numpy()
# r_neg  = torch.clamp(r_use[~y_use], 0, hi).cpu().numpy()
# plt.figure(figsize=(7,4))
# plt.hist(r_all, bins=60, density=True, alpha=0.35, color='gray', label='all')
# if r_pos.size: plt.hist(r_pos, bins=60, density=True, alpha=0.45, label='TRUE mask')
# if r_neg.size: plt.hist(r_neg, bins=60, density=True, alpha=0.45, label='outside TRUE mask')
# plt.axvline(tau_q,    color='k', ls='--', lw=1.2, label=f'tau_q={tau_q:.3f}')
# plt.axvline(tau_star, color='r', ls='-.', lw=1.2, label=f'tau_F1={tau_star:.3f}')
# plt.xlabel('ratio = d2 / d1'); plt.ylabel('density'); plt.title('Overall-mask threshold from ratio')
# plt.legend(); plt.tight_layout(); plt.show()
# # [overall mask] tau_q=0.0000  tau_F1=0.0853 (F1=0.429)