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
import json
import time
import random
import traceback
from copy import deepcopy

import numpy as np
import SimpleITK as sitk
import torch
import portpy.photon as pp

from portpy.ai.models.networks3d import BeamletUNet3D as UNet3D
from portpy.ai.data.beamletdose3d_dataset import BeamletDose3DDataset
from portpy.ai.data.beamletdose3d_dataset import collate_with_meta
from types import SimpleNamespace


def get_ct_image(ct: pp.CT):
    ct_arr = ct.ct_dict['ct_hu_3d'][0]
    ct_image = sitk.GetImageFromArray(ct_arr)
    ct_image.SetOrigin(ct.ct_dict['origin_xyz_mm'])
    ct_image.SetSpacing(ct.ct_dict['resolution_xyz_mm'])
    ct_image.SetDirection(ct.ct_dict['direction'])
    return ct_image

def make_sitk_image_from_array_on_fixed_grid(arr_zyx, origin_xyz_mm, spacing_xyz_mm, direction_xyz=None):
    img = sitk.GetImageFromArray(arr_zyx.astype(np.float32))
    img.SetOrigin(tuple(float(v) for v in origin_xyz_mm))
    img.SetSpacing(tuple(float(v) for v in spacing_xyz_mm))
    if direction_xyz is None:
        direction_xyz = np.eye(3).reshape(-1).tolist()
    img.SetDirection(direction_xyz)
    return img

def resample_pred_to_ct_grid(pred_zyx, ct_img, origin_xyz_mm, spacing_xyz_mm):
    pred_img = make_sitk_image_from_array_on_fixed_grid(
        pred_zyx,
        origin_xyz_mm=origin_xyz_mm,
        spacing_xyz_mm=spacing_xyz_mm,
        direction_xyz=np.eye(3).reshape(-1).tolist()
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    pred_full_img = resampler.Execute(pred_img)
    return sitk.GetArrayFromImage(pred_full_img).astype(np.float32)

def deterministic_beam_ids(data, patient_id):
    data.patient_id = patient_id
    meta_data = data.load_metadata()
    keys = meta_data['beams'].keys()
    values_lists = meta_data['beams'].values()
    list_of_dicts = [dict(zip(keys, values)) for values in zip(*values_lists)]

    num_random_angles = 6
    rng = random.Random(42 + int(patient_id.split("_")[-1]))
    start_angle = rng.choice(list(range(0, 60, 5)))
    random_gantry_angles = [(start_angle + 60 * k) % 360 for k in range(num_random_angles)]
    random_gantry_angles = sorted(random_gantry_angles)

    beam_ids = data.filter_beams_by_properties(
        beams_metadata=list_of_dicts,
        gantry_angles=random_gantry_angles
    )
    return beam_ids

def load_unet_model(ckpt_path, device):
    model = UNet3D(in_channels=4, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model

def build_predicted_influence_matrix_for_patient(
    patient_id,
    data_root,
    split,
    ct_img,
    inf_matrix,
    model,
    device,
    batch_size=4,
    num_workers=0,
    local_data_dir=None,  # add this parameter for on-the-fly preprocessing
):
    """Build the predicted influence matrix A_pred for a single patient using the provided model.
    :param patient_id: The ID of the patient to process.
    :param data_root: The root directory where the patient data is stored.
    :param split: The data split (e.g., "train", "val", "test") to use.
    :param ct_img: The SimpleITK image of the patient's CT scan.
    :param inf_matrix: The original influence matrix object for the patient, used for shape information and coordinate mapping.
    :param model: The trained PyTorch model to use for predictions.
    :param device: The torch device to run the model on (e.g., "cuda" or "cpu").
    :param batch_size: The batch size to use for processing beamlet samples.
    :param num_workers: The number of worker processes to use for data loading.
    :param local_data_dir: Optional local directory to use for on-the-fly preprocessing if beamlet data is missing. This allows the function to preprocess the necessary data if it hasn't been preprocessed yet, without requiring the caller to handle that logic.

    """

    # Minimal on-the-fly preprocessing if data not found
    p_dir = os.path.join(data_root, split, patient_id)
    beamlets_dir = os.path.join(p_dir, "beamlets")

    if not (os.path.isdir(beamlets_dir) and len(os.listdir(beamlets_dir)) > 0):
        print(f"Beamlet data missing for {patient_id}. Creating on the fly...")

        from portpy.ai.preprocess.beamletdose_preprocess import beamletdose_preprocess

        beamletdose_preprocess(
            patient_ids=[patient_id],
            local_data_dir=local_data_dir,  # reuse your existing variable
            out_root=os.path.join(data_root, split),
        )
    ds_opt = SimpleNamespace(
        dataroot=data_root,
        phase=split,
    )
    ds = BeamletDose3DDataset(ds_opt, patient_id=patient_id)

    print(f"{patient_id}: {len(ds)} beamlet samples")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_with_meta
    )

    n_vox, n_cols = inf_matrix.A.shape
    A_pred = np.zeros((n_vox, n_cols), dtype=np.float32)

    with torch.no_grad():
        for batch in dl:
            xb = batch["A"].to(device, non_blocking=True)
            hot_mask = batch["HOT_MASK"].to(device, non_blocking=True)
            scatter_mask = batch["SCATTER_MASK"].to(device, non_blocking=True)
            meta = batch["meta"]

            preds = model(xb)

            mask = torch.maximum(hot_mask, scatter_mask)

            for k in range(preds.size(0)):
                col = int(meta["col"][k])

                y_max = 0.7
                unscale = y_max / 10.0

                pred3 = preds[k, 0].detach().cpu().numpy()
                pred3 = (pred3 * unscale).astype(np.float32)

                m3 = mask[k, 0].detach().cpu().numpy()
                pred3 = pred3 * m3

                origin_xyz_mm = meta["origin_xyz_mm"][k]
                spacing_xyz_mm = meta["spacing_xyz_mm"][k]
                size_xyz = meta["size_xyz"][k]

                if hasattr(origin_xyz_mm, "cpu"):
                    origin_xyz_mm = origin_xyz_mm.cpu().numpy()
                if hasattr(spacing_xyz_mm, "cpu"):
                    spacing_xyz_mm = spacing_xyz_mm.cpu().numpy()
                if hasattr(size_xyz, "cpu"):
                    size_xyz = size_xyz.cpu().numpy()

                origin_xyz_mm = np.asarray(origin_xyz_mm, dtype=np.float32).ravel()
                spacing_xyz_mm = np.asarray(spacing_xyz_mm, dtype=np.float32).ravel()
                size_xyz = np.asarray(size_xyz).astype(int).ravel()

                expected_zyx = (int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0]))
                if tuple(pred3.shape) != expected_zyx:
                    raise ValueError(
                        f"Prediction shape {pred3.shape} != expected {expected_zyx} for {patient_id}, col={col}"
                    )

                pred_full_3d = resample_pred_to_ct_grid(
                    pred_zyx=pred3,
                    ct_img=ct_img,
                    origin_xyz_mm=origin_xyz_mm,
                    spacing_xyz_mm=spacing_xyz_mm,
                )

                pred_col_1d = inf_matrix.dose_3d_to_1d(dose_3d=pred_full_3d).astype(np.float32)
                nz_idx = np.nonzero(pred_col_1d)[0]
                if nz_idx.size:
                    A_pred[nz_idx, col] = pred_col_1d[nz_idx]
    # with torch.no_grad():
    #     for xb, yb, meta in dl:
    #         keep_idx = []
    #         for k in range(len(yb)):
    #             pid = meta["patient_id"][k]
    #             pid = pid.item() if hasattr(pid, "item") else pid
    #             if str(pid) == patient_id:
    #                 keep_idx.append(k)
    #
    #         if not keep_idx:
    #             continue
    #
    #         keep_idx = torch.as_tensor(keep_idx, dtype=torch.long)
    #         xb = xb.index_select(0, keep_idx).to(device, non_blocking=True)
    #         sub_meta = {k: [meta[k][i] for i in keep_idx.tolist()] for k in meta.keys()}
    #
    #         preds = model(xb[:, 0:4, ...])
    #
    #         hot_mask = xb[:, 4:5, ...]
    #         scatter_mask = xb[:, 5:6, ...]
    #         mask = torch.maximum(hot_mask, scatter_mask)
    #
    #         for k in range(preds.size(0)):
    #             col = int(sub_meta["col"][k])
    #
    #             y_max = 0.7
    #             unscale = y_max / 10.0
    #
    #             pred3 = preds[k, 0].detach().cpu().numpy()
    #             pred3 = (pred3 * unscale).astype(np.float32)
    #
    #             m3 = mask[k, 0].detach().cpu().numpy()
    #             pred3 = pred3 * m3
    #
    #             origin_xyz_mm = sub_meta["origin_xyz_mm"][k]
    #             spacing_xyz_mm = sub_meta["spacing_xyz_mm"][k]
    #             size_xyz = sub_meta["size_xyz"][k]
    #
    #             if hasattr(origin_xyz_mm, "cpu"):
    #                 origin_xyz_mm = origin_xyz_mm.cpu().numpy()
    #             if hasattr(spacing_xyz_mm, "cpu"):
    #                 spacing_xyz_mm = spacing_xyz_mm.cpu().numpy()
    #             if hasattr(size_xyz, "cpu"):
    #                 size_xyz = size_xyz.cpu().numpy()
    #
    #             origin_xyz_mm = np.asarray(origin_xyz_mm, dtype=np.float32).ravel()
    #             spacing_xyz_mm = np.asarray(spacing_xyz_mm, dtype=np.float32).ravel()
    #             size_xyz = np.asarray(size_xyz).astype(int).ravel()
    #
    #             expected_zyx = (int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0]))
    #             if tuple(pred3.shape) != expected_zyx:
    #                 raise ValueError(
    #                     f"Prediction shape {pred3.shape} != expected {expected_zyx} for {patient_id}, col={col}"
    #                 )
    #
    #             pred_full_3d = resample_pred_to_ct_grid(
    #                 pred_zyx=pred3,
    #                 ct_img=ct_img,
    #                 origin_xyz_mm=origin_xyz_mm,
    #                 spacing_xyz_mm=spacing_xyz_mm,
    #             )
    #
    #             pred_col_1d = inf_matrix.dose_3d_to_1d(dose_3d=pred_full_3d).astype(np.float32)
    #             nz_idx = np.nonzero(pred_col_1d)[0]
    #             if nz_idx.size:
    #                 A_pred[nz_idx, col] = pred_col_1d[nz_idx]

    A_pred[A_pred < 0] = 0.0
    return A_pred

def main():
    in_dir = "/data"
    data_root = "/inf_data_d1d2_patients_fix_sampling_rev1"
    split = "val"   # change to test if needed
    plan_name = "d1d2_72_beams_16batch_lungprostate_100pat"
    ckpt_path = f"/echo/research/inf_matrix_pred/{plan_name}/checkpoints/ckpt-epoch=004.ckpt"
    out_root = f"echo/research/inf_matrix_pred/inf_matrix_pred_results_batch/{plan_name}/{split}"
    os.makedirs(out_root, exist_ok=True)

    patients = [
        p for p in os.listdir(os.path.join(data_root, split))
        if os.path.isdir(os.path.join(data_root, split, p))
    ]
    patients = sorted(patients)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_unet_model(ckpt_path, device)

    for patient_id in patients:
        try:
            print(f"\n=== Building A_pred for {patient_id} ===")
            p_out = os.path.join(out_root, patient_id)
            os.makedirs(p_out, exist_ok=True)

            save_path = os.path.join(p_out, f"{patient_id}_A_pred.npy")
            if os.path.exists(save_path):
                print(f"Skipping {patient_id}, already exists")
                continue

            data = pp.DataExplorer(data_dir=in_dir)
            data.patient_id = patient_id

            beam_ids = deterministic_beam_ids(data, patient_id)
            ct = pp.CT(data)
            structs = pp.Structures(data)
            beams = pp.Beams(data, beam_ids=beam_ids, load_inf_matrix_full=False)
            inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams, is_full=False)
            ct_img = get_ct_image(ct)

            t0 = time.time()
            A_pred = build_predicted_influence_matrix_for_patient(
                patient_id=patient_id,
                data_root=data_root,
                split=split,
                ct_img=ct_img,
                inf_matrix=inf_matrix,
                model=model,
                device=device,
                batch_size=16,
                num_workers=4,
            )
            np.save(save_path, A_pred)
            print(f"Saved {save_path}")
            print(f"Time: {time.time() - t0:.2f} sec")

        except Exception as e:
            traceback.print_exc()
            print(f"Failed for {patient_id}: {e}")

if __name__ == "__main__":
    main()