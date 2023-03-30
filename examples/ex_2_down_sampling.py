"""
### This example demonstrates performing the following tasks using portpy:

1. Query the existing patients in the database
    (you first need to download the patient database from the link provided in the GitHub page).
2. Query the data provided for a specified patient in the database.
3. Create a simple IMRT plan using CVXPy package. You can call different opensource/commercial optimization engines
    from CVXPy,but you first need to download them and obtain an appropriate license.
    Most commercial optimization engines (e.g., Mosek, Gorubi) give free academic license if you have .edu email
    address
4. Visualise the plan based on down sampled beamlets and voxels (dose_1d distribution, fluence)
5. Evaluate the plan based on down sampled beamlets and voxels


"""

import numpy as np
import portpy.photon as pp
import matplotlib.pyplot as plt

# ### Create plan
# Create my_plan object for the planner beams.
# For the customized beams_dict, you can pass the argument beam_ids
# e.g. my_plan = pp.Plan(patient_name, beam_ids=[0,10,20,30,40,50,60])
data_dir = r'../data'
patient_id = 'Lung_Phantom_Patient_1'
my_plan = pp.Plan(patient_id, data_dir=data_dir)

# PortPy can down-sample beamlets as factor of original finest beamlet resolution.
# e.g if the finest beamlet resolution is 2.5mm then down sampled beamlet resolution can be 5, 7.5, 10mm..
# Example create a influence matrix down sampled beamlets of width and height 5mm
beamlet_down_sample_factor = 4
beamlet_width_mm = my_plan.inf_matrix.beamlet_width_mm * beamlet_down_sample_factor
beamlet_height_mm = my_plan.inf_matrix.beamlet_height_mm * beamlet_down_sample_factor
inf_matrix_db = my_plan.create_inf_matrix(beamlet_width_mm=beamlet_width_mm, beamlet_height_mm=beamlet_height_mm)

# PortPy can down-sample optimization voxels as factor of ct voxels.
# Example: create another influence matrix for down sampled voxels combining 7 ct voxels in x,y direction and 1 ct voxel in z direction.
# It can be done by passing the argument opt_vox_xyz_res_mm = ct_res_xyz * down_sample_factor
down_sample_factor = [7, 7, 1]
opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(my_plan.get_ct_res_xyz_mm(), down_sample_factor)]
inf_matrix_dv = my_plan.create_inf_matrix(opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

# Now, let us also down sample both voxels and beamlets
inf_matrix_dbv = my_plan.create_inf_matrix(beamlet_width_mm=beamlet_width_mm, beamlet_height_mm=beamlet_height_mm,
                                           opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

# Let us create rinds for creating reasonable dose fall off for the plan

rind_max_dose = np.array([1.1, 1.05, 0.9, 0.85, 0.75]) * my_plan.get_prescription()
rind_params = [{'rind_name': 'RIND_0', 'ref_structure': 'PTV', 'margin_start_mm': 0, 'margin_end_mm': 5,
                'max_dose_gy': rind_max_dose[0]},
               {'rind_name': 'RIND_1', 'ref_structure': 'PTV', 'margin_start_mm': 5, 'margin_end_mm': 10,
                'max_dose_gy': rind_max_dose[1]},
               {'rind_name': 'RIND_2', 'ref_structure': 'PTV', 'margin_start_mm': 10, 'margin_end_mm': 30,
                'max_dose_gy': rind_max_dose[2]},
               {'rind_name': 'RIND_3', 'ref_structure': 'PTV', 'margin_start_mm': 30, 'margin_end_mm': 60,
                'max_dose_gy': rind_max_dose[3]},
               {'rind_name': 'RIND_4', 'ref_structure': 'PTV', 'margin_start_mm': 60, 'margin_end_mm': 'inf',
                'max_dose_gy': rind_max_dose[4]}]
my_plan.add_rinds(rind_params=rind_params)

# set rind voxel index for other influence matrices
rinds = [rind for idx, rind in enumerate(my_plan.structures.structures_dict['name']) if 'RIND' in rind]
for rind in rinds:
    inf_matrix_db.set_opt_voxel_idx(my_plan, structure_name=rind)
    inf_matrix_dv.set_opt_voxel_idx(my_plan, structure_name=rind)
    inf_matrix_dbv.set_opt_voxel_idx(my_plan, structure_name=rind)

# ### Run Optimization
# - Run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol dictionary
# - CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
# - For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
# - For mixed integer programs, GUROBI/CPLEX are good choices
# - If you have .edu email address, you can get free academic license for commercial solvers
# - We recommend the commercial solver MOSEK as your solver for the problems in this example,
#   however, if you don't have a license, you can try opensource/free solver SCS or ECOS
#   see [cvxpy](https://www.cvxpy.org/tutorial/advanced/index.html) for more info about CVXPy solvers
# - To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
# create cvxpy problem with max and mean dose clinical criteria and the above objective functions
prob = pp.CvxPyProb(my_plan)
prob.solve(solver='MOSEK', verbose=False)
sol_orig = prob.get_sol()

# running optimization using downsampled beamlets
# create cvxpy problem with max and mean dose clinical criteria and the above objective functions
prob = pp.CvxPyProb(my_plan, inf_matrix=inf_matrix_db)
prob.solve(solver='MOSEK', verbose=False)
sol_db = prob.get_sol()

# running optimization using downsampled voxels
# create cvxpy problem with max and mean dose clinical criteria and the above objective functions
prob = pp.CvxPyProb(my_plan, inf_matrix=inf_matrix_dv)
prob.solve(solver='MOSEK', verbose=False)
sol_dv = prob.get_sol()

# running optimization using downsampled beamlets and voxels
# create cvxpy problem with max and mean dose clinical criteria and the above objective functions
prob = pp.CvxPyProb(my_plan, inf_matrix=inf_matrix_dbv)
prob.solve(solver='MOSEK', verbose=False)
sol_dbv = prob.get_sol()

# # Comment/Uncomment these lines to save & load plan and optimal solutions
pp.save_plan(my_plan, path=r'C:\temp\db4_dv661')
pp.save_optimal_sol(sol_orig, sol_name='sol_orig', path=r'C:\temp\db4_dv661')
pp.save_optimal_sol(sol_db, sol_name='sol_db', path=r'C:\temp\db4_dv661')
pp.save_optimal_sol(sol_dv, sol_name='sol_dv', path=r'C:\temp\db4_dv661')
pp.save_optimal_sol(sol_dbv, sol_name='sol_dbv', path=r'C:\temp\db4_dv661')
my_plan = pp.load_plan('my_plan', path=r'C:\temp\db4_dv661')
sol_orig = pp.load_optimal_sol('sol_orig', path=r'C:\temp\db4_dv661')
sol_db = pp.load_optimal_sol('sol_db', path=r'C:\temp\db4_dv661')
sol_dv = pp.load_optimal_sol('sol_dv', path=r'C:\temp\db4_dv661')
sol_dbv = pp.load_optimal_sol('sol_dbv', path=r'C:\temp\db4_dv661')

# To know the cost of down sampling beamlets, lets compare the dvh of down sampled beamlets with original
structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

fig, ax = plt.subplots(figsize=(12, 8))
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_orig, structs=structs, style='solid', ax=ax)
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_db, structs=structs, style='dotted', ax=ax)
ax.set_title('Cost of Down-Sampling Beamlets  - Original .. Down-Sampled beamlets')
plt.show()

# Similarly to analyze the cost of down sampling voxels, lets compare the dvh of down sampled voxels with original
structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
sol_dv_new = pp.sol_change_inf_matrix(sol_dv, inf_matrix=sol_orig['inf_matrix'])
fig, ax = plt.subplots(figsize=(12, 8))
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_orig, structs=structs, style='solid', ax=ax)
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_dv_new, structs=structs, style='dotted', ax=ax)
ax.set_title('Cost of Down-Sampling Voxels  - Original .. Down-Sampled Voxels')
plt.show()

# To get the discrepancy due to down sampling voxels
fig, ax = plt.subplots(figsize=(12, 8))
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_dv_new, structs=structs, style='solid', ax=ax)
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_dv, structs=structs, style='dotted', ax=ax)
ax.set_title(
    'Discrepancy due to Down-Sampling Voxels  \n - Down sampled with original influence matrix \n .. Down sampled without original influence matrix')
plt.show()

# Now let us plot dvh for analyzing the combined cost of down-sampling beamlets and voxels
structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
sol_dbv_new = pp.sol_change_inf_matrix(sol_dbv, inf_matrix=sol_orig['inf_matrix'])
fig, ax = plt.subplots(figsize=(12, 8))
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_orig, structs=structs, style='solid', ax=ax)
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_dbv_new, structs=structs, style='dotted', ax=ax)
ax.set_title('Combined Cost of Down-Sampling Beamlets and Voxels  \n - Original .. Down-Sampled Beamlets and Voxels')
plt.show()

# Similarly let us plot dvh for analyzing the combined discrepancy of down-sampling beamlets and voxels
structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
fig, ax = plt.subplots(figsize=(12, 8))
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_dbv, structs=structs, style='solid', ax=ax)
ax = pp.Visualize.plot_dvh(my_plan, sol=sol_dbv_new, structs=structs, style='dotted', ax=ax)
ax.set_title(
    'Discrepancy due to Down-Sampling Bemalets and Voxels  \n - Down sampled with original influence matrix \n .. Down sampled without original influence matrix')
plt.show()


