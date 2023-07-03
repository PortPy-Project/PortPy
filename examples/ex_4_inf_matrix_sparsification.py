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
import matplotlib.pyplot as plt


def ex_4_inf_matrix_sparsification():
    """
    1) generating a plan using the sparse matrix (default matrix in PortPy)

    """
    # Create plan_sparse object
    # By default, load_inf_matrix_full=False, and it only loads the sparse matrix
    data_dir = r'../data'
    data = pp.DataExplorer(data_dir=data_dir)
    patient_id = 'Lung_Patient_7'
    data.patient_id = patient_id

    # Load ct, structure and beams as an object
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)

    # create rinds based upon rind definition in optimization params
    opt_params = data.load_config_opt_params(protocol_name='Lung_2Gy_30Fx')
    structs.create_opt_structures(opt_params)

    # load influence matrix based upon beams and structure set
    inf_matrix_sparse = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # load clinical criteria from the config files for which plan to be optimized
    protocol_name = 'Lung_2Gy_30Fx'
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name)

    # Create my_plan object which would load and store all the data needed for optimization
    plan_sparse = pp.Plan(ct, structs, beams, inf_matrix_sparse, clinical_criteria)

    # create cvxpy problem using the clinical criteria and optimization parameters
    opt = pp.Optimization(plan_sparse, opt_params=opt_params)
    opt.create_cvxpy_problem()

    sol_sparse = opt.solve(solver='MOSEK', verbose=True)

    # Calculate the dose using the sparse matrix
    dose_sparse_1d = plan_sparse.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_sparse.get_num_of_fractions())

    """
    3) calculating the full dose for the plan using the full matrix
    
    """
    # Note: It is often computationally impractical to use the full matrix for optimization. We just use the
    #   full matrix to calculate the dose for the solution obtained by sparse matrix and show the resultant discrepancy

    # create plan_full object by specifying load_inf_matrix_full=True
    beams_full = pp.Beams(data, load_inf_matrix_full=True)
    # load influence matrix based upon beams and structure set
    inf_matrix_full = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_full, is_full=True)
    plan_full = pp.Plan(ct, structs, beams, inf_matrix_full, clinical_criteria)
    # use the full influence matrix to calculate the dose for the plan obtained by sparse matrix
    dose_full_1d = plan_full.inf_matrix.A @ (sol_sparse['optimal_intensity'] * plan_full.get_num_of_fractions())

    # Visualize the DVH discrepancy
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(plan_sparse, dose_1d=dose_sparse_1d, struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
    ax = pp.Visualization.plot_dvh(plan_full, dose_1d=dose_full_1d, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True)
    ax.set_title('- Sparse .. Full')
    plt.show()
    print('Done')

    """ 
    3) manually calculating the sparse matrix from the full matrix
    
    """
    # The sparse and full matrices are both pre-calculated and included in PorPy data.
    #   The sparse matrix; however, was obtained by simply zeroing out the small elements in the full matrix that were
    #   less than a threshold specified in "my_plan.inf_matrix.sparse_tol". Here, we manually generate the sparse
    #   matrix from the full matrix using this threshold to clarify the process

    #  Get A_sparse and A_full
    A_full = plan_full.inf_matrix.A
    A_sparse = plan_sparse.inf_matrix.A
    # Get the threshold value used by PortPy to truncate the matrix
    # sparse tol is 1% of the maximum of influence matrix of planner beams
    sparse_tol = plan_sparse.inf_matrix.sparse_tol
    # sparse_tol = 0.01*np.amax(A_full)

    # Truncate the full matrix
    A_full[A_full <= sparse_tol] = 0
    test = np.abs(A_full - A_sparse.todense()) <= 1e-3
    # Check if both influence matrices agree
    assert test.all()


if __name__ == "__main__":
    ex_4_inf_matrix_sparsification()
