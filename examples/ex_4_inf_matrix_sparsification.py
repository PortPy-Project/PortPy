import portpy as pp
import numpy as np
from copy import deepcopy


def ex_4_inf_matrix_sparsification():
    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'F:\Research\Data_newformat\Python-PORT\Data'
    # display the existing patients. To display it in browser rather than console, turn on in_browser=True
    pp.Visualize.display_patients(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, )
    patient_id = 'Lung_Patient_1'
    pp.Visualize.display_patient_metadata(patient_id, data_dir=data_dir)

    # create my_plan object with full influence matrix for the planner beams
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = pp.Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    plan_full = pp.Plan(patient_id, load_inf_matrix_full=True)

    plan_sparse = pp.Plan(patient_id)  # create plan with sparse matrix
    A_sparse = deepcopy(plan_sparse.inf_matrix.A)  # deepcopy so it doesnt modify the object
    A_sparse = A_sparse.todense()  # convert sparse to dense

    # Creating sparse from full influence matrix
    A_full = deepcopy(plan_full.inf_matrix.A)  # deepcopy so it doesnt modify the object
    sparse_tol = plan_sparse.inf_matrix.sparse_tol
    A_full[A_full <= sparse_tol] = 0  # set all the values in full matrix to zero which are less than sparse_tol
    test = np.abs(A_full - A_sparse) <= 1e-3
    assert test.all()  # check if both the influence matrix are similar. if True, then both are similar

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    sol_sparse = pp.Optimize.run_IMRT_fluence_map_CVXPy(plan_sparse, solver='MOSEK')

    # above solution is obtained using the sparse influence matrix

    # sol_full = pp.sol_change_inf_matrix(sol, inf_matrix_full)  # create a new solution with full influence matrix
    # my_plan = pp.Plan.load_plan(path=r'C:\temp')
    # sol_sparse = pp.load_optimal_sol('sol_sparse', path=r'C:\temp')
    # sol_full = pp.load_optimal_sol('sol_full', path=r'C:\temp')
    # my_plan.save_plan(path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_sparse, sol_name='sol_sparse', path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_full, sol_name='sol_full', path=r'C:\temp')

    # Now, let us compare the dvh using sparse and full influence matrix
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    dose_sparse_1d = plan_sparse.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_sparse.get_num_of_fractions())  # getting dose in 1d for sparse matrix
    dose_full_1d = plan_full.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_full.get_num_of_fractions())  # getting dose in 1d for full matrix
    pp.Visualize.plot_dvh(plan_sparse, dose_1d=dose_sparse_1d, structs=structs, style='solid', show=False, norm_flag=True) # plot dvh using the above dose
    pp.Visualize.plot_dvh(plan_full, dose_1d=dose_full_1d, structs=structs, style='dotted', create_fig=False, norm_flag=True)
    print('Done')


if __name__ == "__main__":
    ex_4_inf_matrix_sparsification()
