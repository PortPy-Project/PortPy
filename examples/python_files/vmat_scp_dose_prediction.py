import os

import numpy as np

import portpy.photon as pp
import os
import matplotlib.pyplot as plt
from portpy.ai.preprocess.predict_using_model import predict_using_model

# change directory to portpy ai module to preprocess the portpy data, train and test the model
os.chdir('../../portpy/ai')

# # preprocess portpy data
in_dir = r'../../data'
out_dir = r'../../ai_data'

# os.system('python ./preprocess/data_preprocess.py --in_dir ../../data --out_dir ../../ai_data')

# Train preprocess data
# os.system('python train.py --dataroot ../../ai_data --netG unet_128 --name portpy_test_2 --model doseprediction3d --direction AtoB --lambda_L1 1 --dataset_mode dosepred3d --norm batch --batch_size 1 --pool_size 0 --display_port 8097 --lr 0.0002 --input_nc 8 --output_nc 1 --display_freq 10 --print_freq 1 --gpu_ids 0')

# Test preprocess data
# os.system('python test.py --dataroot ../../ai_data --netG unet_128 --name portpy_test_2 --phase test --mode eval --model doseprediction3d --input_nc 8 --output_nc 1 --direction AtoB --dataset_mode dosepred3d --norm batch')

# predicted dose back to portpy
patient_id = 'Lung_Patient_6'
model_name = 'portpy_test_2'
pred_dose = predict_using_model(patient_id=patient_id, in_dir=in_dir, model_name=model_name)

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

arcs_dict = {'arcs': [{'arc_id': "01", "control_point_ids": beam_ids[0:int(len(beam_ids) / 2)]},
                          {'arc_id': "02", "control_point_ids": beam_ids[int(len(beam_ids) / 2):]}]}
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
