import portpy as pp
import numpy as np
from copy import deepcopy


def ex_5_inf_matrix_sparsification():
    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'F:\Research\Data_newformat\Python-PORT\Data'
    # display the existing patients. To display it in browser rather than console, turn on in_browser=True
    pp.Visualize.display_patients(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, )
    patient_name = 'Lung_Patient_1'
    pp.Visualize.display_patient_metadata(patient_name, data_dir=data_dir)

    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = pp.Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = pp.Plan(patient_name, load_inf_matrix_full=True)

    inf_matrix_sparse = my_plan.inf_matrix  # getting influence  matrix
    A_sparse = deepcopy(inf_matrix_sparse.A)  # deepcopy so it doesnt modify the object
    A_sparse = A_sparse.todense()  # convert sparse to dense

    # Creating sparse from full influence matrix
    inf_matrix_full = my_plan.create_inf_matrix(is_sparse=False)  # creating object with full influence matrix
    A_full = deepcopy(inf_matrix_full.A)  # deepcopy so it doesnt modify the object
    A_full[A_full <= inf_matrix_sparse.sparse_tol] = 0  # set all the values in full matrix to zero which are less than sparse_tol
    test = np.abs(A_full - A_sparse) <= 1e-3
    assert test.all()  # check if both the influence matrix are similar. if True, then both are similar
    # assert np.allclose(A_full, A_sparse)  # check if both the influence matrix are similar. if True, then both are similar

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    sol = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, solver='MOSEK')

    # above solution is obtained using the sparse influence matrix
    # Now, let us compare the dvh using sparse and full influence matrix
    sol_full = pp.sol_change_inf_matrix(sol, inf_matrix_full)  # create a new solution with full influence matrix

    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    pp.Visualize.plot_dvh(my_plan, sol=sol, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_full, structs=structs, style='dotted', create_fig=False)


if __name__ == "__main__":
    ex_5_inf_matrix_sparsification()
