"""
### PortPy provides full dense influence matrix (i.e., including all scattering components) and a truncated sparse
    version (used for computational efficiency). This example clarifies the relationship and differences between
    these matrices by showing the followings:
    1- generating a plan using the sparse matrix (default matrix in PortPy)
    2- calculating the full dose for the plan using the full matrix
    3- manually calculating the sparse matrix from the full matrix

"""

import portpy.photon as pp
import numpy as np
from copy import deepcopy


def ex_4_inf_matrix_sparsification():

    # ***************** 1) generating a plan using the sparse matrix (default matrix in PortPy)************************
    # Create plan_sparse object
    # By default, load_inf_matrix_full=False, and it only loads the spase matrix
    data_dir = r'../../data'
    patient_id = 'Lung_Phantom_Patient_1'
    plan_sparse = pp.Plan(patient_id, data_dir=data_dir)
    sol_sparse = pp.Optimize.run_IMRT_fluence_map_CVXPy(plan_sparse, solver='MOSEK')
    # Calculate the dose using the sparse matrix
    dose_sparse_1d = plan_sparse.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_sparse.get_num_of_fractions())

    # ***************** 2) calculating the full dose for the plan using the full matrix***********************
    # Note: It is often computationally impractical to use the full matrix for optimization. We just use the
    #   full matrix to calculate the dose for the solution obtained by sparse matrix and show the resultant discrepancy

    # create plan_full object by specifying load_inf_matrix_full=True
    plan_full = pp.Plan(patient_id, load_inf_matrix_full=True)
    # use the full influence matrix to calculate the dose for the plan obtained by sparse matrix
    dose_full_1d = plan_full.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_full.get_num_of_fractions())
    # Visualize the DVH discrepancy
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    pp.Visualize.plot_dvh(plan_sparse, dose_1d=dose_sparse_1d, structs=structs, style='solid', show=False, norm_flag=True)
    pp.Visualize.plot_dvh(plan_full, dose_1d=dose_full_1d, structs=structs, style='dotted', create_fig=False, norm_flag=True)
    print('Done')

    # ***************** 3) manually calculating the sparse matrix from the full matrix***********************
    # The sparse and full matrices are both pre-calculated and included in PorPy data.
    #   The sparse matrix; however, was obtained by simply zeroing out the small elements in the full matrix that were
    #   less than a threshold specified in "my_plan.inf_matrix.sparse_tol". Here, we manually generate the sparse
    #   matrix from the full matrix using this threshold to clarify the process

    #  Get A_sparse and A_full
    A_full = plan_full.inf_matrix.A
    A_sparse = plan_sparse.inf_matrix.A
    # Get the threshold value used by PortPy to truncate the matrix
    sparse_tol = plan_sparse.inf_matrix.sparse_tol
    # Truncate the full matrix
    A_full[A_full <= sparse_tol] = 0
    test = np.abs(A_full - A_sparse.todense()) <= 1e-3
    # Check if both influence matrices agree
    assert test.all()


if __name__ == "__main__":
    ex_4_inf_matrix_sparsification()
