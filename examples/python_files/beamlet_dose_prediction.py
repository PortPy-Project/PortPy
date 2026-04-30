import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import portpy.photon as pp

from portpy.ai.preprocess.beamletdose_preprocess import beamletdose_preprocess
from portpy.ai.preprocess.build_apred_batch import build_predicted_influence_matrix_for_patient
from portpy.ai.train import train
from portpy.ai.test import test


# -----------------------------
# 1. Preprocess beamlet data
# -----------------------------
in_dir = "../../data"
beamlet_data_root = "../../ai_beamlet_data"
checkpoints_dir = "../../checkpoints"
results_dir = "../../results"

patients = [f"Lung_Patient_{i}" for i in range(2, 101)]
# add prostate patients if desired:
# patients += [f"Prostate_Patient_{i}" for i in range(1, 130)]

# below script will download the necessary metadata and preprocess the beamlet dose data from hugging face for the selected patients, saving the processed data to `beamlet_ai_data` directory.
beamletdose_preprocess(
    patient_ids=patients,
    meta_stage_root="../../PortPy_MetadataOnly",
    data_stage_root="../../PortPy_Dataset_SelectedBeams",
    out_root=beamlet_data_root,
)
#OR if you have already have the data downloaded for selected patients, you can preprocess and save the data to `beamlet_ai_data` directory using the same function above but with `meta_stage_root` and `data_stage_root` pointing to your local directories where the metadata and data are stored.
# beamletdose_preprocess(
#     patient_ids=patients,
#     local_data_dir=in_dir,
#     out_root=beamlet_data_root)

# Manually split:
# beamlet_ai_data/train/Lung_Patient_x
# beamlet_ai_data/val/Lung_Patient_x
# beamlet_ai_data/test/Lung_Patient_x


# -----------------------------
# 2. Train beamlet model
# -----------------------------
train_options = {
    "dataroot": beamlet_data_root,
    "checkpoints_dir": checkpoints_dir,
    "name": "portpy_beamlet_unet2",
    "model": "beamletdose3d",
    "dataset_mode": "beamletdose3d",
    "netG": "beamlet_unet",
    "input_nc": 4,
    "output_nc": 1,
    "norm": "batch",
    "batch_size": 2,
    "num_threads": 2,
    "lr": 1e-4,
    "n_epochs": 100,
    "n_epochs_decay": 0,
    "display_id": -1,
    "display_freq": 100,
    "print_freq": 10,
    "gpu_ids": [0],
    "val_phase": "val",
    "lr_policy": "plateau",
    "lambda_bg": 0.0005,
}

train(train_options)


# -----------------------------
# 3. Test beamlet model
# -----------------------------
test_options = {
    "dataroot": beamlet_data_root,
    "checkpoints_dir": checkpoints_dir,
    "results_dir": results_dir,
    "name": "portpy_beamlet_unet",
    "model": "beamletdose3d",
    "dataset_mode": "beamletdose3d",
    "netG": "beamlet_unet",
    "input_nc": 4,
    "output_nc": 1,
    "phase": "test",
    "eval": True,
    "norm": "batch",
    "lambda_bg": 0.0005,
    "epoch": "best_mae",   # <-- add this
    "test_task": "beamlet_dose_prediction",
}

test(test_options)


# -----------------------------
# 4. Infer A_pred for one patient
# -----------------------------
patient_id = "Lung_Patient_94"
protocol_name = "Lung_2Gy_30Fx"

data = pp.DataExplorer(data_dir=in_dir)
data.patient_id = patient_id

ct = pp.CT(data)
structs = pp.Structures(data)

# IMPORTANT: use same deterministic beam selection used in preprocessing/build_apred
from portpy.ai.preprocess.build_apred_batch import deterministic_beam_ids, get_ct_image, load_unet_model

beam_ids = deterministic_beam_ids(data, patient_id)
beams = pp.Beams(data, beam_ids=beam_ids, load_inf_matrix_full=True)
inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams, is_full=True)
ct_img = get_ct_image(ct)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = os.path.join(checkpoints_dir, "portpy_beamlet_unet", "best_mae_net_G.pth")
model = load_unet_model(ckpt_path, device)

A_pred = build_predicted_influence_matrix_for_patient(
    patient_id=patient_id,
    data_root=beamlet_data_root,
    split="test",
    ct_img=ct_img,
    inf_matrix=inf_matrix, # it just this infleunce matrix object is used to get the dimensions and metadata for building A_pred, the actual values in inf_matrix.A are not used and can be either the sparse or full matrix.
    model=model,
    device=device,
    batch_size=2,
    num_workers=0,
)


# -----------------------------
# 5. Optimize using predicted beamlet matrix
# -----------------------------
from copy import deepcopy
from scipy import sparse

clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
opt_params = data.load_config_opt_params(protocol_name=protocol_name)

A_gt = deepcopy(inf_matrix.A).astype(np.float32)

# Sparse thresholded actual matrix.
thr_gt = 0.01 * np.max(A_gt)
S_gt = sparse.csr_matrix(np.where(A_gt >= thr_gt, A_gt, 0.0))

# Sparse thresholded predicted matrix.
thr_pred = 0.01 * np.max(A_pred) if np.max(A_pred) > 0 else 0.0
S_pred = sparse.csr_matrix(np.where(A_pred >= thr_pred, A_pred, 0.0))

# Solve using predicted beamlet matrix.
inf_matrix.A = S_pred
plan_pred = pp.Plan(
    ct=ct,
    structs=structs,
    beams=beams,
    inf_matrix=inf_matrix,
    clinical_criteria=clinical_criteria,
)
opt_pred = pp.Optimization(plan_pred, opt_params=opt_params, clinical_criteria=clinical_criteria)
opt_pred.create_cvxpy_problem()
sol_pred = opt_pred.solve(solver="MOSEK", verbose=False)
x_pred = sol_pred["optimal_intensity"] * plan_pred.get_num_of_fractions()


# -----------------------------
# 6. DVH visualization
# -----------------------------
# -----------------------------
# 6. Compare DVH: A_full @ x_actual vs A_full @ x_pred
# Solve using actual/full beamlet matrix
inf_matrix.A = S_gt
plan_gt = pp.Plan(
    ct=ct,
    structs=structs,
    beams=beams,
    inf_matrix=inf_matrix,
    clinical_criteria=clinical_criteria,
)
opt_gt = pp.Optimization(plan_gt, opt_params=opt_params, clinical_criteria=clinical_criteria)
opt_gt.create_cvxpy_problem()
sol_gt = opt_gt.solve(solver="MOSEK", verbose=False)
x_actual = sol_gt["optimal_intensity"] * plan_gt.get_num_of_fractions()

# Fair comparison: evaluate both fluence maps on actual/full matrix
dose_gt_1d = A_gt @ x_actual      # A_full @ x_actual
dose_pred_1d = A_gt @ x_pred      # A_full @ x_pred

if "lung" in patient_id.lower():
    struct_names = ["PTV", "ESOPHAGUS", "HEART", "CORD", "LUNG_L", "LUNG_R", "LUNGS_NOT_GTV"]
else:
    struct_names = ["PTV", "BLADDER", "RECTUM", "FEMUR_L", "FEMUR_R"]

fig, ax = plt.subplots(figsize=(12, 8))

ax = pp.Visualization.plot_dvh(
    plan_gt,
    dose_1d=dose_gt_1d,
    struct_names=struct_names,
    style="solid",
    ax=ax,
    norm_flag=True,
)

ax = pp.Visualization.plot_dvh(
    plan_gt,
    dose_1d=dose_pred_1d,
    struct_names=struct_names,
    style="dotted",
    ax=ax,
    norm_flag=True,
)

ax.set_title("DVH comparison: GT vs pred")
plt.show()