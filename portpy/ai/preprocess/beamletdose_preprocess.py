# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use — including offering PortPy as a service,
# or incorporating it into a commercial product — requires
# a separate commercial license.
# ----------------------------------------------------------------------

import os

import random
import traceback

import numpy as np
import torch
import SimpleITK as sitk
import portpy.photon as pp


# keep your existing helper imports
from portpy.ai.preprocess.beamlet_preprocess_utils import (
    get_body_center_xy_mm,
    make_reference_image_from_center,
    resample_ct_to_fixed_iso_box,
    resample_array_on_ct_grid_to_fixed_iso_box,
    resample_masked_array_on_ct_grid_to_fixed_iso_box,
    precompute_beam_geometry_on_fixed_grid,
    compute_d1_d2_from_precomputed_geom,
    build_fast_dose_1d_to_3d_mapper,
    fast_dose_1d_to_3d,
    split_d1_in_out,
)

# BEV (ray-aligned) grid. Imported from the INFERENCE package on purpose: this is
# the exact geometry RayUNetPredictor uses, so training samples and prediction-time
# inputs are built by the same code and cannot drift apart.
from portpy.ai.inference.ray_geometry import (
    get_rotation_matrix as ray_get_rotation_matrix,
    get_ct_sitk_image,
    resample_to_grid,
    estimate_body_entry_distance_mm,
    build_ray_grid_with_beam_axes,
    compute_ray_patch_d1_d2_split,
    source_lps_from_beam,
    beamlet_center_lps,
)

# # -----------------------------
# # USER SETTINGS
# # -----------------------------
HF_REPO_ID = "PortPy-Project/PortPy_Dataset"
HF_REPO_TYPE = "dataset"

#
UINT16_MAX = 65535

# -----------------------------
# HELPERS
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def resolve_data_dir(root):
    """
    PortPy DataExplorer usually wants the directory that directly contains patient folders.
    If snapshot_download created a nested 'data/' folder, use that.
    """
    data_subdir = os.path.join(root, "data")
    return data_subdir if os.path.isdir(data_subdir) else root

def download_metadata_only(patient_id: str, meta_stage_root: str):
    """
    Download only lightweight metadata files for one patient.
    """
    allow_patterns = [
        f"data/{patient_id}/CT_MetaData.json",
        f"data/{patient_id}/StructureSet_MetaData.json",
        f"data/{patient_id}/OptimizationVoxels_MetaData.json",
        f"data/{patient_id}/PlannerBeams.json",
        f"data/{patient_id}/Beams/*_MetaData.json",
    ]
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Please install huggingface_hub to use this script: pip install huggingface_hub")
        pass
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=meta_stage_root,
        allow_patterns=allow_patterns
    )

def get_beam_ids_for_patient(patient_id: str, meta_stage_root: str, gantry_angles):
    """
    Download metadata only, then use the same PortPy metadata filtering logic
    as your original script to get beam ids at runtime.
    """
    download_metadata_only(patient_id, meta_stage_root)

    meta_data_dir = resolve_data_dir(meta_stage_root)
    data_meta = pp.DataExplorer(data_dir=meta_data_dir)
    data_meta.patient_id = patient_id
    meta_data = data_meta.load_metadata()

    keys = meta_data["beams"].keys()
    values_lists = meta_data["beams"].values()
    list_of_dicts = [dict(zip(keys, values)) for values in zip(*values_lists)]

    beam_ids = data_meta.filter_beams_by_properties(
        beams_metadata=list_of_dicts,
        gantry_angles=gantry_angles
    )
    return beam_ids

def download_selected_beams(patient_id: str, beam_ids, data_stage_root: str):
    """
    Download the actual patient data for only the selected beam ids.
    """
    if len(beam_ids) == 0:
        raise ValueError(f"No beam ids selected for {patient_id}")

    # PortPy docs support a beam-id list here
    pp.download_portpy_data(
        [patient_id],
        out=data_stage_root,
        beam_ids=[int(b) for b in beam_ids],
        beam_mode = "ids"
    )

def quantize_mm_to_uint16(arr, max_mm=2000.0):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, max_mm)
    q = np.round(arr * (UINT16_MAX / max_mm)).astype(np.uint16)
    return q


def dequantize_uint16_to_mm(arr_q, max_mm=2000.0):
    arr_q = np.asarray(arr_q, dtype=np.float32)
    arr = arr_q * (max_mm / UINT16_MAX)
    return arr

def beamletdose_preprocess(
        patient_ids,
        meta_stage_root=None,
        data_stage_root=None,
        out_root=None,
        local_data_dir=None,
        target_spacing_xyz_mm=(2.5, 2.5, 2.5),
        target_size_xyz=(224, 192, 96),
        use_patient_level_ct_box=True,
        d_mm_max=2000.0,
        grid="patient",
        bev_view_size_mm=(512.0, 128.0, 128.0),
        bev_out_size=(256, 64, 64)):
    """
    Write per-beamlet training samples for the beamlet dose model.

    Two sampling grids are available and they produce the same npz fields, so the same
    dataset, model and training loop are used either way:

    - ``grid='patient'`` (default): one world-aligned box per patient
      (``target_size_xyz`` at ``target_spacing_xyz_mm``), shared by every beamlet. The
      beam direction is carried only by the d1/d2 channels. The CT is written once per
      patient as ``ct.npz``.
    - ``grid='bev'``: one ray-aligned patch PER BEAMLET, with its x axis along the
      source->beamlet ray, so the network sees the dose in its natural frame. The grid
      is built by ``portpy.ai.inference.ray_geometry`` -- the same code
      ``RayUNetPredictor`` uses at inference time -- so training inputs and prediction
      inputs cannot drift apart. The CT differs per beamlet and is stored in each
      sample (clipped int16 HU) instead of once per patient. The d1/d2/d1_out distance
      channels are NOT stored: they are pure functions of the patch geometry and are
      recomputed identically by the dataset and by the predictor via
      ``compute_ray_patch_d1_d2_split``.

    :param patient_ids: list of PortPy patient ids
    :param meta_stage_root: staging dir for the metadata-only Hugging Face download
    :param data_stage_root: staging dir for the selected-beam Hugging Face download
    :param out_root: output root; one folder per patient is created under it
    :param local_data_dir: use already-downloaded PortPy data instead of Hugging Face
    :param target_spacing_xyz_mm: patient-grid voxel size, mm (``grid='patient'``)
    :param target_size_xyz: patient-grid size in voxels (``grid='patient'``)
    :param use_patient_level_ct_box: share one CT box across beams (``grid='patient'``)
    :param d_mm_max: full scale of the uint16 quantization used for d1/d2, mm
    :param grid: ``'patient'`` or ``'bev'``
    :param bev_view_size_mm: BEV patch extent (depth, width, height), mm. Default
        matches the ray-UNet configuration used in
        ``examples/python_files/inf_matrix_portpy_beams_voxels_cross_eval.py``.
    :param bev_out_size: BEV patch size in voxels (Nx, Ny, Nz), x along the ray. Each
        entry must be divisible by 8 (the ray UNet uses num_levels=3). The default
        (256, 64, 64) over (512, 128, 128) mm gives 2 mm isotropic voxels.
    :return: None; samples are written under ``out_root``

    :Example:

    >>> beamletdose_preprocess(patient_ids=['Lung_Patient_94'],
    ...                        local_data_dir='../../data',
    ...                        out_root='../../ai_beamlet_data',
    ...                        grid='bev')
    """
    # -----------------------------
    # MAIN
    # -----------------------------
    if out_root is None:
        raise ValueError("out_root must be provided.")

    if grid not in ("patient", "bev"):
        raise ValueError("grid must be 'patient' or 'bev', got {!r}".format(grid))

    if grid == "bev" and any(int(n) % 8 for n in bev_out_size):
        raise ValueError(
            "bev_out_size {} must be divisible by 8 on every axis (the ray UNet runs "
            "num_levels=3, i.e. 3 pooling stages).".format(tuple(bev_out_size))
        )

    if meta_stage_root is not None:
        ensure_dir(meta_stage_root)
    if data_stage_root is not None:
        ensure_dir(data_stage_root)
    ensure_dir(out_root)

    for pid in patient_ids:
        try:
            print(f"\n== Patient {pid} ==")

            # -----------------------------
            # STEP 1: discover beam_ids from metadata only
            # -----------------------------
            # num_random_angles = 6
            # rng = random.Random(42 + int(pid.split("_")[-1]))
            # random_gantry_angles = sorted(rng.sample(list(range(0, 360, 5)), num_random_angles))
            print(f"\n== Patient {pid} ==")

            num_random_angles = 6
            rng = random.Random(42 + int(pid.split("_")[-1]))

            start_angle = rng.choice(list(range(0, 60, 5)))
            random_gantry_angles = [(start_angle + 60 * k) % 360 for k in range(num_random_angles)]
            random_gantry_angles = sorted(random_gantry_angles)

            print(f"Random gantry angles for {pid}: {random_gantry_angles}")

            if local_data_dir is not None:
                # Use already-downloaded local PortPy data.
                data_dir = resolve_data_dir(local_data_dir)
                data = pp.DataExplorer(data_dir=data_dir)
                data.patient_id = pid
                meta_data = data.load_metadata()

                keys = meta_data["beams"].keys()
                values_lists = meta_data["beams"].values()
                list_of_dicts = [dict(zip(keys, values)) for values in zip(*values_lists)]

                beam_ids = data.filter_beams_by_properties(
                    beams_metadata=list_of_dicts,
                    gantry_angles=random_gantry_angles,
                )

            else:
                # Existing Hugging Face path.
                if meta_stage_root is None or data_stage_root is None:
                    raise ValueError(
                        "Either provide local_data_dir, or provide both meta_stage_root and data_stage_root."
                    )

                beam_ids = get_beam_ids_for_patient(
                    patient_id=pid,
                    meta_stage_root=meta_stage_root,
                    gantry_angles=random_gantry_angles,
                )

                download_selected_beams(
                    patient_id=pid,
                    beam_ids=beam_ids,
                    data_stage_root=data_stage_root,
                )

                data_dir = resolve_data_dir(data_stage_root)
                data = pp.DataExplorer(data_dir=data_dir)
                data.patient_id = pid
                meta_data = data.load_metadata()

            print(f"Selected beam ids for {pid}: {beam_ids}")

            # IMPORTANT: use the same runtime-selected beam_ids
            ct = pp.CT(data)
            # ct_arr_orig = ct.ct_dict["ct_hu_3d"][0]
            structs = pp.Structures(data)

            body_cx_mm, body_cy_mm = get_body_center_xy_mm(structs, ct)

            beams = pp.Beams(data, beam_ids=beam_ids, load_inf_matrix_full=True)
            inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams, is_full=True)

            # vox_coords = inf_matrix.get_voxel_coordinates()
            valid_1d = np.ones(inf_matrix.A.shape[0], dtype=np.float32)
            valid_3d = inf_matrix.dose_1d_to_3d(dose_1d=valid_1d)
            dose_mapper = build_fast_dose_1d_to_3d_mapper(inf_matrix)

            if grid == "bev":
                # Built once per patient: the ray patches resample straight off the
                # CT grid, and the BODY entry ray-march reuses the cached array.
                ct_sitk = get_ct_sitk_image(ct)
                body_mask_ct_zyx = structs.get_structure_mask_3d("BODY").astype(np.uint8)
                body_sitk = sitk.GetImageFromArray(body_mask_ct_zyx)
                body_sitk.CopyInformation(ct_sitk)

            # local output dirs
            p_dir = os.path.join(out_root, pid)
            beams_dir = os.path.join(p_dir, "beams")
            beamlets_dir = os.path.join(p_dir, "beamlets")

            ensure_dir(p_dir)
            ensure_dir(beams_dir)
            ensure_dir(beamlets_dir)

            # -----------------------------
            # Save patient CT once if enabled
            # (BEV has a different CT crop per beamlet, so it is stored per sample)
            # -----------------------------
            if use_patient_level_ct_box and grid != "bev":
                first_beam_id = inf_matrix.beamlets_dict[0]["beam_id"]
                first_iso_list = beams.get_iso_center(first_beam_id)

                patient_center = np.asarray(
                    [body_cx_mm, body_cy_mm, first_iso_list["z_mm"]],
                    dtype=np.float32
                )

                ct_ref_img, ct_ref_meta = make_reference_image_from_center(
                    center_xyz_mm=patient_center,
                    spacing_xyz_mm=target_spacing_xyz_mm,
                    size_xyz=target_size_xyz,
                    direction=np.eye(3).reshape(-1).tolist()
                )

                ct_down = resample_ct_to_fixed_iso_box(
                    ct,
                    ct_ref_img,
                    default_ct_value=-1000.0
                )
                # ct_torch_tensor = torch.from_numpy(ct_down)
                #
                # ct_path = os.path.join(p_dir, "ct.pt")
                # torch.save(
                #     {
                #         "ct": ct_torch_tensor,
                #         "origin_xyz_mm": ct_ref_meta["origin_xyz_mm"],
                #         "spacing_xyz_mm": ct_ref_meta["spacing_xyz_mm"],
                #         "size_xyz": ct_ref_meta["size_xyz"],
                #         "center_xyz_mm": ct_ref_meta["center_xyz_mm"],
                #     },
                #     ct_path,
                # )
                ct_path = os.path.join(p_dir, "ct.npz")
                np.savez_compressed(
                    ct_path,
                    ct=ct_down.astype(np.float32),
                    origin_xyz_mm=np.asarray(ct_ref_meta["origin_xyz_mm"], dtype=np.float32),
                    spacing_xyz_mm=np.asarray(ct_ref_meta["spacing_xyz_mm"], dtype=np.float32),
                    size_xyz=np.asarray(ct_ref_meta["size_xyz"], dtype=np.int16),
                    center_xyz_mm=np.asarray(ct_ref_meta["center_xyz_mm"], dtype=np.float32),
                )

            # -----------------------------
            # Loop over selected beams
            # -----------------------------
            for i in range(len(inf_matrix.beamlets_dict)):
                print("Beam #:", i)

                beam_id = inf_matrix.beamlets_dict[i]["beam_id"]
                gantry = beams.get_gantry_angle(beam_id)
                SAD_mm = beams.beams_dict["SAD_mm"][i]
                SSD_mm = beams.beams_dict["SSD_mm"][i]

                couch = 0
                collimator = None

                iso_list = beams.get_iso_center(beam_id)
                iso = np.asarray(
                    [iso_list["x_mm"], iso_list["y_mm"], iso_list["z_mm"]],
                    dtype=np.float32
                )

                if grid == "bev":
                    # One ray-aligned patch per BEAMLET, built with the SAME functions
                    # RayUNetPredictor calls at inference time (portpy.ai.inference.
                    # ray_geometry), so training inputs and prediction inputs match.
                    R = ray_get_rotation_matrix(gantry, couch, 0.0 if collimator is None
                                                else float(collimator))
                    source = source_lps_from_beam(iso, R, SAD_mm)

                    start_beamlet = inf_matrix.beamlets_dict[i]["start_beamlet_idx"]
                    end_beamlet = inf_matrix.beamlets_dict[i]["end_beamlet_idx"] + 1
                    print("start beamlet is:", start_beamlet)
                    print("end beamlet is:", end_beamlet)

                    sx = float(bev_view_size_mm[0]) / int(bev_out_size[0])
                    iso_index = int(bev_out_size[0]) // 2

                    for col in range(start_beamlet, end_beamlet):
                        if col % 100 == 0:
                            print("Beamlet #:", col)

                        local_idx = col - start_beamlet
                        b_x = float(np.asarray(
                            inf_matrix.beamlets_dict[i]["position_x_mm"][0][local_idx]).squeeze())
                        b_y = float(np.asarray(
                            inf_matrix.beamlets_dict[i]["position_y_mm"][0][local_idx]).squeeze())

                        center = beamlet_center_lps(b_x, b_y, R, iso)

                        # Put the beamlet/isocentre plane at the middle of the patch,
                        # never starting behind the source.
                        den = float(np.linalg.norm(center - source))
                        x0_mm = max(0.0, den - iso_index * sx)

                        origin, spacing, direction = build_ray_grid_with_beam_axes(
                            source_lps=source,
                            target_lps=center,
                            R_lps_to_bcs=R,
                            view_size_mm=bev_view_size_mm,
                            out_size=bev_out_size,
                            x0_mm=x0_mm,
                        )

                        ct_patch = sitk.GetArrayFromImage(resample_to_grid(
                            ct_sitk, origin, spacing, direction, bev_out_size,
                            default_value=-1000.0, interp=sitk.sitkLinear)).astype(np.float32)

                        body_patch = (sitk.GetArrayFromImage(resample_to_grid(
                            body_sitk, origin, spacing, direction, bev_out_size,
                            default_value=0.0,
                            interp=sitk.sitkNearestNeighbor)) > 0.5).astype(np.uint8)

                        beamlet_values = inf_matrix.A[:, col]
                        if not isinstance(beamlet_values, np.ndarray):
                            beamlet_values = beamlet_values.toarray().ravel()
                        beamlet_values = np.asarray(beamlet_values, dtype=np.float32).ravel().copy()
                        max_val = float(np.max(beamlet_values))
                        if max_val == 0:
                            print(f"Warning: beamlet {col} has all-zero dose values.")
                        else:
                            # Drop the 0.1%-of-peak fog, as the research preprocess does.
                            beamlet_values[beamlet_values <= 0.001 * max_val] = 0.0

                        beamlet_3d = fast_dose_1d_to_3d(beamlet_values, dose_mapper)
                        dose_sitk = sitk.GetImageFromArray(beamlet_3d.astype(np.float32))
                        dose_sitk.CopyInformation(ct_sitk)
                        target_patch = sitk.GetArrayFromImage(resample_to_grid(
                            dose_sitk, origin, spacing, direction, bev_out_size,
                            default_value=0.0, interp=sitk.sitkLinear)).astype(np.float32)

                        t_entry = estimate_body_entry_distance_mm(
                            body_sitk, source, center, mask_zyx=body_mask_ct_zyx)

                        # d1/d2/d1_out are pure functions of (view, out, x0, t_entry,
                        # body) and are recomputed identically in the dataset and in
                        # RayUNetPredictor, so they are not stored.
                        bl_path = os.path.join(beamlets_dir, f"bl_beam{beam_id}_col{col}.npz")
                        np.savez_compressed(
                            bl_path,
                            patient_id=np.array(pid),
                            beam_id=np.int32(beam_id),
                            col=np.int32(col),
                            local_idx=np.int32(local_idx),
                            gantry=np.float32(gantry),
                            sad_mm=np.float32(SAD_mm),
                            ssd_mm=np.float32(SSD_mm),
                            # CT kept as float32 HU. The research preprocess stored a
                            # rounded int16 instead; that costs half the disk but makes
                            # the normalized CT channel differ from what
                            # RayUNetPredictor computes by ~1.4e-4 (0.5 HU x the RED
                            # slope), so the training input would not be bit-identical
                            # to the inference input.
                            ct=ct_patch.astype(np.float32),
                            body_mask=body_patch,
                            target=target_patch,
                            # ray-patch geometry: everything needed to rebuild the grid,
                            # recompute the distance channels, and back-project the output
                            origin_xyz_mm=np.asarray(origin, dtype=np.float32),
                            spacing_xyz_mm=np.asarray(spacing, dtype=np.float32),
                            size_xyz=np.asarray(bev_out_size, dtype=np.int16),
                            direction_3x3=np.asarray(direction, dtype=np.float32),
                            view_size_mm=np.asarray(bev_view_size_mm, dtype=np.float32),
                            out_size=np.asarray(bev_out_size, dtype=np.int16),
                            x0_mm=np.float32(x0_mm),
                            t_entry=np.float32(0.0 if t_entry is None else t_entry),
                            d_mm_max=np.float32(d_mm_max),
                            # beam / beamlet geometry
                            iso_xyz_mm=np.asarray(iso, dtype=np.float32),
                            source_lps=np.asarray(source, dtype=np.float32),
                            beamlet_center_lps=np.asarray(center, dtype=np.float32),
                            beamlet_center_x_mm=np.float32(b_x),
                            beamlet_center_y_mm=np.float32(b_y),
                        )

                    print("##############################################################")
                    print("                       END OF BEAM", i)
                    print("##############################################################")
                    continue

                beam_center = np.asarray([body_cx_mm, body_cy_mm, iso[2]], dtype=np.float32)

                ref_img, ref_meta = make_reference_image_from_center(
                    center_xyz_mm=beam_center,
                    spacing_xyz_mm=target_spacing_xyz_mm,
                    size_xyz=target_size_xyz,
                    direction=np.eye(3).reshape(-1).tolist()
                )

                valid_3d_down = resample_array_on_ct_grid_to_fixed_iso_box(
                    valid_3d,
                    ct=ct,
                    ref_img=ref_img,
                    is_mask=True,
                    default_value=0.0,
                ).astype(np.float32)

                body_mask_3d = structs.get_structure_mask_3d("BODY").astype(np.float32)
                body_mask_3d_down = resample_array_on_ct_grid_to_fixed_iso_box(
                    body_mask_3d,
                    ct=ct,
                    ref_img=ref_img,
                    is_mask=True,
                    default_value=0.0,
                ).astype(bool)

                beam_geom = precompute_beam_geometry_on_fixed_grid(
                    ref_meta=ref_meta,
                    iso_xyz_mm=iso,
                    gantry=gantry,
                    couch=couch,
                    collimator=collimator,
                    SAD_mm=SAD_mm
                )

                if not use_patient_level_ct_box:
                    ct_down = resample_ct_to_fixed_iso_box(
                        ct,
                        ref_img,
                        default_ct_value=-1000.0
                    )
                    # ct_torch_tensor = torch.from_numpy(ct_down)
                    #
                    # ct_path = os.path.join(p_dir, f"ct_beam_{beam_id}.pt")
                    # torch.save(
                    #     {
                    #         "ct": ct_torch_tensor,
                    #         "origin_xyz_mm": ref_meta["origin_xyz_mm"],
                    #         "spacing_xyz_mm": ref_meta["spacing_xyz_mm"],
                    #         "size_xyz": ref_meta["size_xyz"],
                    #         "iso_xyz_mm": tuple(iso.tolist()),
                    #         "center_xyz_mm": ref_meta["center_xyz_mm"],
                    #     },
                    #     ct_path,
                    # )
                    ct_path = os.path.join(p_dir, f"ct_beam_{beam_id}.npz")
                    np.savez_compressed(
                        ct_path,
                        ct=ct_down.astype(np.float32),
                        origin_xyz_mm=np.asarray(ref_meta["origin_xyz_mm"], dtype=np.float32),
                        spacing_xyz_mm=np.asarray(ref_meta["spacing_xyz_mm"], dtype=np.float32),
                        size_xyz=np.asarray(ref_meta["size_xyz"], dtype=np.int16),
                        iso_xyz_mm=np.asarray(iso, dtype=np.float32),
                        center_xyz_mm=np.asarray(ref_meta["center_xyz_mm"], dtype=np.float32),
                    )

                start_beamlet = inf_matrix.beamlets_dict[i]["start_beamlet_idx"]
                end_beamlet = inf_matrix.beamlets_dict[i]["end_beamlet_idx"] + 1
                print("start beamlet is:", start_beamlet)
                print("end beamlet is:", end_beamlet)

                # open-beam dose for this beam
                beam_mask = np.zeros((inf_matrix.A.shape[1]), dtype=np.float32)
                beam_mask[start_beamlet:end_beamlet] = 1.0

                beams_1d = inf_matrix.A @ beam_mask
                if not isinstance(beams_1d, np.ndarray):
                    beams_1d = np.asarray(beams_1d).ravel()

                beams_3d = inf_matrix.dose_1d_to_3d(dose_1d=beams_1d)
                beams_3d_down, _ = resample_masked_array_on_ct_grid_to_fixed_iso_box(
                    beams_3d,
                    valid_mask_zyx=valid_3d,
                    ct=ct,
                    ref_img=ref_img,
                    default_value=0.0,
                )

                # beams_3d_torch_tensor = torch.from_numpy(beams_3d_down)
                #
                # beam_path = os.path.join(beams_dir, f"beam_{beam_id}_open.pt")
                # torch.save(
                #     {
                #         "beam_id": beam_id,
                #         "gantry": gantry,
                #         "sad_mm": SAD_mm,
                #         "ssd_mm": SSD_mm,
                #         "iso_xyz_mm": tuple(iso.tolist()),
                #         "center_xyz_mm": ref_meta["center_xyz_mm"],
                #         "origin_xyz_mm": ref_meta["origin_xyz_mm"],
                #         "spacing_xyz_mm": ref_meta["spacing_xyz_mm"],
                #         "size_xyz": ref_meta["size_xyz"],
                #         "open_beam_dose": beams_3d_torch_tensor,
                #     },
                #     beam_path,
                # )
                beam_path = os.path.join(beams_dir, f"beam_{beam_id}_open.npz")
                np.savez_compressed(
                    beam_path,
                    beam_id=np.int32(beam_id),
                    gantry=np.float32(gantry),
                    sad_mm=np.float32(SAD_mm),
                    ssd_mm=np.float32(SSD_mm),
                    iso_xyz_mm=np.asarray(iso, dtype=np.float32),
                    center_xyz_mm=np.asarray(ref_meta["center_xyz_mm"], dtype=np.float32),
                    origin_xyz_mm=np.asarray(ref_meta["origin_xyz_mm"], dtype=np.float32),
                    spacing_xyz_mm=np.asarray(ref_meta["spacing_xyz_mm"], dtype=np.float32),
                    size_xyz=np.asarray(ref_meta["size_xyz"], dtype=np.int16),
                    open_beam_dose=beams_3d_down.astype(np.float32),
                )

                print("-----------------------------------------------------")

                for col in range(start_beamlet, end_beamlet):
                    if col % 100 == 0:
                        print("Beamlet #:", col)

                    beamlet_values = inf_matrix.A[:, col]
                    if not isinstance(beamlet_values, np.ndarray):
                        beamlet_values = beamlet_values.toarray().ravel()

                    beamlet_3d = fast_dose_1d_to_3d(beamlet_values, dose_mapper)
                    beamlet_3d_down, _ = resample_masked_array_on_ct_grid_to_fixed_iso_box(
                        beamlet_3d,
                        valid_mask_zyx=valid_3d,
                        ct=ct,
                        ref_img=ref_img,
                        default_value=0.0,
                    )
                    # beamlet_torch_tensor = torch.from_numpy(beamlet_3d_down)

                    local_idx = col - start_beamlet
                    b_x = inf_matrix.beamlets_dict[i]["position_x_mm"][0][local_idx].squeeze()
                    b_y = inf_matrix.beamlets_dict[i]["position_y_mm"][0][local_idx].squeeze()

                    d1_3d_down, d2_3d_down = compute_d1_d2_from_precomputed_geom(
                        precomp_geom=beam_geom,
                        b_x=b_x,
                        b_y=b_y,
                        SAD_mm=SAD_mm
                    )

                    d1_out_3d_down, d1_in_3d_down, t_entry = split_d1_in_out(
                        d1_3d_down=d1_3d_down,
                        d2_3d_down=d2_3d_down,
                        body_mask_down=body_mask_3d_down,
                        ray_radius_mm=3.0,
                    )
                    d1_q = quantize_mm_to_uint16(d1_3d_down)
                    d2_q = quantize_mm_to_uint16(d2_3d_down)

                    bl_path = os.path.join(beamlets_dir, f"bl_beam{beam_id}_col{col}.npz")
                    np.savez_compressed(
                        bl_path,
                        patient_id=np.array(pid),
                        beam_id=np.int32(beam_id),
                        col=np.int32(col),
                        gantry=np.float32(gantry),
                        center_xyz_mm=np.asarray(ref_meta["center_xyz_mm"], dtype=np.float32),
                        origin_xyz_mm=np.asarray(ref_meta["origin_xyz_mm"], dtype=np.float32),
                        spacing_xyz_mm=np.asarray(ref_meta["spacing_xyz_mm"], dtype=np.float32),
                        size_xyz=np.asarray(ref_meta["size_xyz"], dtype=np.int16),
                        d_mm_max=np.float32(d_mm_max),
                        d1=d1_q.astype(np.uint16),
                        d2=d2_q.astype(np.uint16),
                        t_entry=np.float32(0.0 if t_entry is None else t_entry),
                        body_mask=body_mask_3d_down.astype(np.uint8),
                        target=beamlet_3d_down.astype(np.float32),
                    )
                    # d1_torch_tensor = torch.from_numpy(d1_q)
                    # d2_torch_tensor = torch.from_numpy(d2_q)
                    #
                    # body_mask_torch_tensor = torch.from_numpy(body_mask_3d_down.astype(np.uint8))
                    # # d1_out_torch_tensor = torch.from_numpy(d1_out_3d_down)
                    #
                    # bl_path = os.path.join(beamlets_dir, f"bl_beam{beam_id}_col{col}.pt")
                    # torch.save(
                    #     {
                    #         "patient_id": pid,
                    #         "beam_id": beam_id,
                    #         "col": col,
                    #         "gantry": gantry,
                    #         "center_xyz_mm": ref_meta["center_xyz_mm"],
                    #         "origin_xyz_mm": ref_meta["origin_xyz_mm"],
                    #         "spacing_xyz_mm": ref_meta["spacing_xyz_mm"],
                    #         "size_xyz": ref_meta["size_xyz"],
                    #         'd_mm_max': D_MM_MAX,
                    #         "d1": d1_torch_tensor,
                    #         "d2": d2_torch_tensor,
                    #         "t_entry": t_entry,
                    #         "body_mask": body_mask_torch_tensor,
                    #         # "d1_out": d1_out_torch_tensor,
                    #         "target": beamlet_torch_tensor,
                    #     },
                    #     bl_path,
                    # )

                print("##############################################################")
                print("                       END OF BEAM", i)
                print("##############################################################")

            print(f"Finished patient {pid}")

        except Exception as e:
            print(f"ERROR processing patient {pid}: {e}")
            traceback.print_exc()
            continue

    print("Done!!!")

if __name__ == "__main__":
    patients = [f"Lung_Patient_{i}" for i in range(2, 101)]

    beamletdose_preprocess(
        patient_ids=patients,
        meta_stage_root="./PortPy_MetadataOnly",
        data_stage_root="./PortPy_Dataset_SelectedBeams",
        out_root="./inf_data_d1d2_patients_fix_sampling_rev1"
    )
    # #OR use below is patients are already downloaded locally.
    # beamletdose_preprocess(
    #     patient_ids=patients,
    #     local_data_dir="../../data",
    #     out_root="../../beamlet_ai_data",
    # )