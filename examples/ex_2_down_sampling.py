"""
    This example demonstrates performing the following tasks using portpy:
    1- Query the existing patients in the database
        (you first need to download the patient database from the link provided in the GitHub page).
    2- Query the data provided for a specified patient in the database.
    3- Create a simple IMRT plan using CVXPy package. You can call different opensource/commercial optimization engines
        from CVXPy,but you first need to download them and obtain an appropriate license.
        Most commercial optimization engines (e.g., Mosek, Gorubi) give free academic license if you have .edu email
        address
    4- Visualise the plan based on down sampled beamlets and voxels (dose_1d distribution, fluence)
    5- Evaluate the plan based on down sampled beamlets and voxels

"""

import portpy as pp


def ex_2_down_sampling():
    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'F:\Research\Data_newformat\Python-PORT\Data'

    # display the existing patients
    pp.Visualize.display_patients(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, )
    patient_id = 'Lung_Patient_1'
    pp.Visualize.display_patient_metadata(patient_id, data_dir=data_dir)

    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = pp.Plan(patient_id)

    # PortPy can down-sample beamlets as factor of original finest beamlet resolution.
    # e.g if the finest beamlet resolution is 2.5mm then down sampled beamlet resolution can be 5, 7.5, 10mm..
    # Example create a influence matrix down sampled beamlets of width and height 5mm
    beamlet_width_mm = my_plan.inf_matrix.beamlet_width_mm * 2
    beamlet_height_mm = my_plan.inf_matrix.beamlet_height_mm * 2
    inf_matrix_db = my_plan.create_inf_matrix(beamlet_width_mm=beamlet_width_mm, beamlet_height_mm=beamlet_height_mm)

    # PortPy can down-sample optimization voxels as factor of ct voxels.
    # Example: create another influence matrix for down sampled voxels combining 5 ct voxels in x,y direction and 1 ct voxel in z direction.
    # It can be done by passing the argument opt_vox_xyz_res_mm = ct_res_xyz * down_sample_factor
    down_sample_factor = [5, 5, 1]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(my_plan.get_ct_res_xyz_mm(), down_sample_factor)]
    inf_matrix_dv = my_plan.create_inf_matrix(opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # Now, let us also down sample both voxels and beamlets
    inf_matrix_dbv = my_plan.create_inf_matrix(beamlet_width_mm=5, beamlet_height_mm=5, opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    sol_orig = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, solver='MOSEK')
    sol_db = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_db)  # optimize using downsampled beamlets
    sol_dv = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_dv)  # optimize using downsampled voxels
    sol_dbv = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_dbv)  # optimize using downsampled beamlets and voxels

    # # Comment/Uncomment these lines to save & load plan and optimal solutions
    my_plan.save_plan(path=r'C:\temp')
    my_plan.save_optimal_sol(sol_orig, sol_name='sol_orig', path=r'C:\temp')
    my_plan.save_optimal_sol(sol_db, sol_name='sol_db', path=r'C:\temp')
    my_plan.save_optimal_sol(sol_dv, sol_name='sol_dv', path=r'C:\temp')
    my_plan.save_optimal_sol(sol_dbv, sol_name='sol_dbv', path=r'C:\temp')
    # my_plan = pp.Plan.load_plan(path=r'C:\temp')
    # sol_1 = pp.load_optimal_sol('sol_1', path=r'C:\temp')
    # sol_2 = pp.load_optimal_sol('sol_2', path=r'C:\temp')
    # sol_3 = pp.load_optimal_sol('sol_3', path=r'C:\temp')

    # plot fluence in 3d and 2d using the arguments beam id and sol generated using optimization
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_orig)
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_db)
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_dv)
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_dbv)

    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_orig)
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_db)
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_dv)
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_dbv)

    # To know the cost of down sampling beamlets, lets compare the dvh of down sampled beamlets with original
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    pp.Visualize.plot_dvh(my_plan, sol=sol_orig, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_db, structs=structs, style='dotted', create_fig=False)

    # To get the discrepancy due to down sampling beamlets
    sol_db_new = pp.sol_change_inf_matrix(sol_db, inf_matrix=sol_orig['inf_matrix'])
    pp.Visualize.plot_dvh(my_plan, sol=sol_db_new, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_db, structs=structs, style='dotted', create_fig=False)

    # Similarly to analyze the cost of down sampling voxels, lets compare the dvh of down sampled voxels with original
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    pp.Visualize.plot_dvh(my_plan, sol=sol_orig, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_dv, structs=structs, style='dotted', create_fig=False)

    # To get the discrepancy due to down sampling voxels
    sol_dv_new = pp.sol_change_inf_matrix(sol_dv, inf_matrix=sol_orig['inf_matrix'])
    pp.Visualize.plot_dvh(my_plan, sol=sol_dv_new, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_dv, structs=structs, style='dotted', create_fig=False)

    # Now let us plot dvh for analyzing the combined cost of down-sampling beamlets and voxels
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    pp.Visualize.plot_dvh(my_plan, sol=sol_orig, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_dbv, structs=structs, style='dotted', create_fig=False)

    # Similarly let us plot dvh for analyzing the combined discrepancy of down-sampling beamlets and voxels
    sol_dbv_new = pp.sol_change_inf_matrix(sol_dbv, inf_matrix=sol_orig['inf_matrix'])
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    pp.Visualize.plot_dvh(my_plan, sol=sol_orig, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_dbv_new, structs=structs, style='dotted', create_fig=False)

    # Visualize dose in 2d axial slice for all the cases
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_orig)
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_db)
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_dv)

    print('Done!')


if __name__ == "__main__":
    ex_2_down_sampling()
