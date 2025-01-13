import os

import portpy.photon as pp
import SimpleITK as sitk
import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from portpy.ai.preprocess.predict_using_model import predict_using_model
from portpy.ai.preprocess.data_preprocess import data_preprocess
from portpy.ai.train import train
from portpy.ai.test import test

# # preprocess portpy data
in_dir = r'../../data'
out_dir = r'../../ai_data'

# ### 2. Training and testing the model
#
# Train the model. You can change the parameters for the training as show below

# Provide only the arguments you want to override
train_options = {
    "dataroot": "../../ai_data",
    "checkpoints_dir": "../../checkpoints",
    "netG": "unet_128",
    "name": "portpy_test_3",
    "model": "doseprediction3d",
    "direction": "AtoB",
    "lambda_L1": 1,
    "dataset_mode": "dosepred3d",
    "norm": "batch",
    "batch_size": 1,
    "pool_size": 0,
    "display_port": 8097,
    "lr": 0.0002,
    "input_nc": 8,
    "output_nc": 1,
    "display_freq": 10,
    "print_freq": 1,
    "gpu_ids": [0]  # Converted to a list since multiple GPUs may be supported
}

train(train_options)  # Run training directly in Jupyter Notebook

# You can uncomment and run below in case if you want to run train script from CLI
#!python ../portpy/ai/train.py --dataroot ../ai_data --netG unet_128 --name portpy_test_3 --model doseprediction3d --direction AtoB --lambda_L1 1 --dataset_mode dosepred3d --norm batch --batch_size 1 --pool_size 0 --display_port 8097 --lr 0.0002 --input_nc 8 --output_nc 1 --display_freq 10 --print_freq 1 --gpu_ids 0


# Test the model
test_options = {
    "dataroot": "../../ai_data",
    "netG": "unet_128",
    "checkpoints_dir": "../../checkpoints",
    "results_dir": "../../results",
    "name": "portpy_test_3",
    "phase": "test",
    "mode": "eval",
    "eval": True,  # Boolean flag
    "model": "doseprediction3d",
    "input_nc": 8,
    "output_nc": 1,
    "direction": "AtoB",
    "dataset_mode": "dosepred3d",
    "norm": "batch"
}
test(test_options)
# !python ../portpy/ai/test.py --dataroot ../ai_data --netG unet_128 --checkpoints_dir ../checkpoints --results_dir ../results --name portpy_test_2 --phase test --mode eval --eval --model doseprediction3d --input_nc 8 --output_nc 1 --direction Ato

# predicted dose back to portpy
patient_id = 'Lung_Patient_6'
model_name = 'portpy_test_3'
pred_dose = predict_using_model(patient_id=patient_id, in_dir=in_dir, out_dir=out_dir, model_name=model_name, checkpoints_dir='../../checkpoints', results_dir='../../results')


# load portpy data
data = pp.DataExplorer(data_dir=in_dir)
data.patient_id = patient_id
# Load ct and structure set for the above patient using CT and Structures class
ct = pp.CT(data)
ct_arr = ct.ct_dict['ct_hu_3d'][0]
structs = pp.Structures(data)

beam_ids = np.arange(37, 72)
beams = pp.Beams(data, beam_ids=beam_ids)

# create rinds based upon rind definition in optimization params
protocol_name = 'Lung_2Gy_30Fx'

# load influence matrix based upon beams and structure set
inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

# load clinical criteria from the config files for which plan to be optimized
clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

pred_dose_1d = inf_matrix.dose_3d_to_1d(dose_3d=pred_dose)

arcs_dict = {'arcs': [{'arc_id': "01", "beam_ids": beam_ids[0:int(len(beam_ids) / 2)]},
                          {'arc_id': "02", "beam_ids": beam_ids[int(len(beam_ids) / 2):]}]}
# Create arcs object using arcs dictionary and influence matrix
arcs = pp.Arcs(arcs_dict=arcs_dict, inf_matrix=inf_matrix)

# create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria, arcs=arcs)

# Loading hyper-parameter values for optimization problem
protocol_name = 'Lung_2Gy_30Fx_vmat'
vmat_opt_params = data.load_config_opt_params(protocol_name=protocol_name)

# Initialize Optimization
vmat_opt = pp.VmatScpOptimization(my_plan=my_plan,
                                  opt_params=vmat_opt_params)
# Run Sequential convex algorithm for optimising the plan.
# The final result will be stored in sol and convergence will store the convergence history (i.e., results of each iteration)
sol, convergence = vmat_opt.run_sequential_cvx_algo_prediction(pred_dose_1d=pred_dose_1d, solver='MOSEK', verbose=True)

pp.save_optimal_sol(sol, sol_name='sol_vmat_pred.pkl', path=os.path.join(r'C:\temp', data.patient_id))
pp.save_obj_as_pickle(convergence, obj_name='convergence_pred.pkl', path=os.path.join(r'C:\temp', data.patient_id))
pp.save_plan(my_plan, plan_name='my_plan_vmat_pred.pkl', path=os.path.join(r'C:\temp', data.patient_id))

struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD','LUNG_L']
fig, ax = plt.subplots(figsize=(12, 8))
ax = pp.Visualization.plot_dvh(my_plan, sol=sol, struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
ax = pp.Visualization.plot_dvh(my_plan, dose_1d=pred_dose_1d, struct_names=struct_names, style='dotted', ax=ax,
                               norm_flag=True)
ax.set_title('- Optimized .. Predicted')
plt.show()
