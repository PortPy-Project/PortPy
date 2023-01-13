"""
    This example demonstrates performing the following tasks using portpy:
    1- Query the existing patients in the database
        (you first need to download the patient database from the link provided in the GitHub page).
    2- Query the data provided for a specified patient in the database.
    3- Create a simple IMRT plan using CVXPy package. You can call different opensource/commercial optimization engines
        from CVXPy,but you first need to download them and obtain an appropriate license.
        Most commercial optimization engines (e.g., Mosek, Gorubi) give free academic license if you have .edu email
        address
    4- Visualise the plan based on down sampled beamlets and voxels (dose distribution, fluence)
    5- Evaluate the plan based on down sampled beamlets and voxels

"""

import portpy as pp


def eg_2_down_sampling():
    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'F:\Research\Data_newformat\Python-PORT\Data'

    # display the existing patients
    pp.Visualize.display_patients(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, )
    patient_name = 'Lung_Patient_1'
    pp.Visualize.display_patient_metadata(patient_name, data_dir=data_dir)

    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = pp.Plan(patient_name)

    # create a influence matrix down sampled beamlets of width and height 5mm
    inf_matrix_2 = my_plan.create_inf_matrix(beamlet_width_mm=5, beamlet_height_mm=5)

    # create another influence matrix for down sampled voxels combining 5 ct voxels in x,y direction and 1 ct voxel in z direction
    inf_matrix_3 = my_plan.create_inf_matrix(down_sample_xyz=[5, 5, 1])

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    sol_1 = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, solver='MOSEK')
    sol_2 = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_2)
    sol_3 = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_3)

    # # Comment/Uncomment these lines to save & load plan and optimal solutions
    my_plan.save_plan(path=r'C:\temp')
    my_plan.save_optimal_sol(sol_3, sol_name='sol_3', path=r'C:\temp')
    my_plan.save_optimal_sol(sol_2, sol_name='sol_2', path=r'C:\temp')
    my_plan.save_optimal_sol(sol_1, sol_name='sol_1', path=r'C:\temp')
    # my_plan = pp.Plan.load_plan(path=r'C:\temp')
    # sol_1 = pp.Plan.load_optimal_sol('sol_1', path=r'C:\temp')
    # sol_2 = pp.Plan.load_optimal_sol('sol_2', path=r'C:\temp')
    # sol_3 = pp.Plan.load_optimal_sol('sol_3', path=r'C:\temp')

    # # plot fluence
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_1)
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_2)
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_3)
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_1)
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_2)
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_3)
    #

    # plot dvh dvh for all the cases
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    pp.Visualize.plot_dvh(my_plan, sol=sol_1, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_2, structs=structs, style='dotted', create_fig=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_3, structs=structs, style='dashed', create_fig=False)

    # Visualize 2d dose for both the cases
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_1)
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_2)
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_3)

    print('Done!')


if __name__ == "__main__":
    eg_2_down_sampling()
