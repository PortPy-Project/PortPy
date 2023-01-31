"""
    This example demonstrates the use of portpy to create down sampled influence matrix and
    optimize the plan by selecting optimal beams using beam angle optimization
"""
import portpy as pp


def ex_6_bao_benchmark():
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
    my_plan = pp.Plan(patient_id, beam_ids=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66])

    # create a influence matrix down sampled beamlets of width and height 5mm
    down_sample_factor = [5, 5, 1]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(my_plan.get_ct_res_xyz_mm(), down_sample_factor)]
    inf_matrix_dbv = my_plan.create_inf_matrix(beamlet_width_mm=5, beamlet_height_mm=5,
                                               opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    pp.save_inf_matrix(inf_matrix=inf_matrix_dbv, inf_name='inf_matrix_dbv', path=r'C:\temp')
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
    sol_bao = pp.Optimize.run_IMRT_fluence_map_CVXPy_BAO_benchmark(my_plan, inf_matrix=inf_matrix_dbv)

    # Comment/Uncomment these lines to save & load plan and optimal solutions
    # my_plan = pp.load_plan(plan_name='plan_bao', path=r'C:\temp')
    # sol_bao = pp.load_optimal_sol('sol_bao', path=r'C:\temp')
    # my_plan.save_plan(plan_name='plan_bao', path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_bao, sol_name='sol_bao', path=r'C:\temp')

    # creating plan for the planner's beams
    plan_planner = pp.Plan(patient_id)
    inf_matrix_planner_dbv = plan_planner.create_inf_matrix(beamlet_width_mm=5, beamlet_height_mm=5,
                                                            opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)
    sol_planner = pp.Optimize.run_IMRT_fluence_map_CVXPy(plan_planner, inf_matrix=inf_matrix_planner_dbv)

    # plot dvh dvh for both the cases
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    pp.Visualize.plot_dvh(my_plan, sol=sol_bao, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(plan_planner, sol=sol_planner, structs=structs, style='dotted', create_fig=False)


if __name__ == "__main__":
    ex_6_bao_benchmark()
