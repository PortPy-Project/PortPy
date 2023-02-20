"""
    This example demonstrates the use of portpy_photon to create down sampled influence matrix and
    optimize the plan by selecting optimal beams using beam angle optimization
"""
import portpy_photon as pp
import numpy as np


def ex_6_boo_benchmark():
    # Enter patient name
    patient_id = 'Lung_Patient_1'

    # visualize patient metadata for beams_dict and structures
    pp.Visualize.display_patient_metadata(patient_id)

    # display patients
    pp.Visualize.display_patients()

    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    # creating plan and select among the beams which are 30 degrees apart
    # beam_ids = [0, 37, 42, 48, 54, 60, 66]
    beam_ids = [0, 6, 12, 18, 24, 30, 37, 42, 48, 54, 60, 66]
    plan_boo = pp.Plan(patient_id, beam_ids=beam_ids)

    # create a influence matrix down sampled beamlets of width and height 5mm
    down_sample_factor = [5, 5, 1]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(plan_boo.get_ct_res_xyz_mm(), down_sample_factor)]
    beamlet_down_sample_factor = 2
    beamlet_width_mm = plan_boo.inf_matrix.beamlet_width_mm * beamlet_down_sample_factor
    beamlet_height_mm = plan_boo.inf_matrix.beamlet_height_mm * beamlet_down_sample_factor
    # inf_matrix_dbv = plan_boo.create_inf_matrix(beamlet_width_mm=beamlet_width_mm, beamlet_height_mm=beamlet_height_mm,
    #                                             opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    # beam angle and fluence map optimization with downscaled influence matrix
    sol_boo = pp.load_optimal_sol('sol_boo_w_sm_10', path=r'C:\temp')
    inf_matrix_dbv = sol_boo['inf_matrix']
    sol_boo = pp.Optimize.run_IMRT_fluence_map_CVXPy_BOO_benchmark(plan_boo, inf_matrix=inf_matrix_dbv,
                                                                   ptv_overdose_weight=1000, ptv_underdose_weight=10000)

    # Comment/Uncomment these lines to save & load plan and optimal solutions
    # plan_boo = pp.load_plan(plan_name='plan_boo', path=r'C:\temp')
    # sol_boo = pp.load_optimal_sol('sol_boo_w_sm_10', path=r'C:\temp')
    # plan_planner = pp.load_plan(plan_name='plan_planner', path=r'C:\temp')
    # sol_planner = pp.load_optimal_sol('sol_planner', path=r'C:\temp')
    pp.save_plan(plan_boo, plan_name='plan_boo', path=r'C:\temp')
    pp.save_optimal_sol(sol_boo, sol_name='sol_boo', path=r'C:\temp')

    # Similarly, let's create plan for the planner's beams
    plan_planner = pp.Plan(patient_id)
    inf_matrix_planner_dbv = plan_planner.create_inf_matrix(beamlet_width_mm=beamlet_width_mm,
                                                            beamlet_height_mm=beamlet_height_mm,
                                                            opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)
    sol_planner = pp.Optimize.run_IMRT_fluence_map_CVXPy(plan_planner, inf_matrix=inf_matrix_planner_dbv)

    # Comment/Uncomment these lines to save & load plan and optimal solutions
    # plan_planner.save_plan(plan_name='plan_planner_no_sm', path=r'C:\temp')
    # plan_planner.save_optimal_sol(sol_planner, sol_name='sol_planner_no_sm', path=r'C:\temp')
    # sol_planner = pp.load_optimal_sol(sol_name='sol_planner_w_sm_10', path=r'C:\temp')

    # Identifying bao and planner gantry angles
    bao_gantry_angles = (np.asarray(plan_boo.beams.beams_dict['gantry_angle']) + 1) * (sol_boo['optimal_beams'] > 0)
    bao_gantry_angles = bao_gantry_angles[bao_gantry_angles > 0] - 1  # add and subtract 1 to check for 0 degree angle
    print('BAO gantry angles: {}'.format(bao_gantry_angles))
    planner_gantry_angles = plan_planner.beams.beams_dict['gantry_angle']
    print('Planner gantry angles: {}'.format(planner_gantry_angles))

    # plot dvh dvh for both the cases
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    pp.Visualize.plot_dvh(plan_boo, sol=sol_boo, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(plan_planner, sol=sol_planner, structs=structs, style='dotted', create_fig=False)


if __name__ == "__main__":
    ex_6_boo_benchmark()
