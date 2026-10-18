"""

This example demonstrates the use of portpy.photon.imrt_scp module to create a step-and-shoot IMRT
plan using sequential convex programming. The aperture shapes and their monitor units are optimized
directly, so every iterate is deliverable and no fluence map is sequenced afterwards.

The plan uses the beams chosen by the human planner, each delivering 10 apertures, with PortPy's
lung clinical criteria and optimization template.

"""

import os
import portpy.photon as pp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def imrt_scp_tutorial():
    """
     1) Accessing the portpy data (DataExplorer class)

     To start using this resource, users are required to download the latest version of the dataset, which can be found at (https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit). Then, the dataset can be accessed as demonstrated below.

    """

    # specify the patient data location.
    data_dir = r'../../data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)
    data.patient_id = 'Lung_Patient_6'

    # Load ct and structure set for the above patient using CT and Structures class
    ct = pp.CT(data)
    structs = pp.Structures(data)

    # with no beam_ids, PortPy uses the beams selected by an expert human planner
    beams = pp.Beams(data)

    # load clinical criteria from the config files for which plan to be optimized
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name='Lung_2Gy_30Fx')

    # Loading hyper-parameter values for optimization problem
    opt_params = data.load_config_opt_params(protocol_name='Lung_2Gy_30Fx_vmat')

    # Creating optimization structures (i.e., Rinds, PTV-GTV)
    structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)

    # Loading influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # scale influence matrix to get correct MU for TPS import
    inf_matrix_scale_factor = opt_params['opt_parameters'].get('inf_matrix_scale_factor', 1)
    inf_matrix.A = inf_matrix.A * np.float32(inf_matrix_scale_factor)

    # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix,
                      clinical_criteria=clinical_criteria)

    """
     2) Creating apertures for each beam (create_imrt_plan)

    """
    # Give every beam its own apertures. Clinical step-and-shoot IMRT delivers roughly 5-15
    # segments per beam, so 10 over 7 beams gives 70 segments in total.
    imrt_plan = pp.create_imrt_plan(my_plan, num_apertures=10)

    """
     3) Optimize the IMRT plan using sequential convex programming

    """
    # If column generation flag is turned on in opt_params, generate initial feasible apertures
    if opt_params['opt_parameters']['initial_leaf_pos'].lower() == 'cg':
        col_gen = pp.ImrtOptimizationColGen(my_plan=imrt_plan, opt_params=opt_params)
        sol_col_gen = col_gen.run_col_gen_algo(solver='MOSEK', verbose=True, accept_unknown=True)

    # Initialize Optimization
    imrt_opt = pp.ImrtScpOptimization(my_plan=imrt_plan, opt_params=opt_params)
    # Run Sequential convex algorithm for optimising the plan.
    # The final result will be stored in sol and convergence will store the convergence history (i.e., results of each iteration)
    convergence = imrt_opt.run_sequential_cvx_algo(solver='MOSEK', verbose=True)
    sol = convergence[imrt_opt.best_iteration]
    sol['inf_matrix'] = imrt_plan.inf_matrix

    # Visualize convergence. The convergence dataframe contains the following columns:
    df = pd.DataFrame(convergence, columns=['outer_iteration', 'inner_iteration', 'step_size_f_b', 'forward_backward', 'intermediate_obj_value', 'actual_obj_value', 'accept'])
    # We can, for example, plot the actual and intermediate objective values against the outer iteration
    df.plot(x='outer_iteration', y=['actual_obj_value', 'intermediate_obj_value'])
    plt.show()

    """
     4) Visualize and evaluate the plan

    """
    # plot dvh for the above structures
    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNGS_NOT_GTV']
    pp.Visualization.plot_dvh(imrt_plan, sol=sol, struct_names=struct_names, style='solid', ax=ax)
    plt.show()

    # Evaluate clinical criteria
    pp.Evaluation.display_clinical_criteria(my_plan=imrt_plan, sol=sol)

    # int_v holds the MU of every aperture; an aperture with no MU is not delivered
    mu = np.asarray(sol['int_v'])
    print('apertures with MU: {} of {}, total MU {:.2f}'.format(int(np.sum(mu > 1e-9)), mu.size,
                                                                mu.sum()))

    # saving and loading the plan for future use (utils)
    # # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(imrt_plan, plan_name='my_plan_imrt_scp.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol, sol_name='sol_imrt_scp.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # # imrt_plan = pp.load_plan(plan_name='my_plan_imrt_scp.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # # sol = pp.load_optimal_sol(sol_name='sol_imrt_scp.pkl', path=os.path.join(r'C:\temp', data.patient_id))


if __name__ == "__main__":
    imrt_scp_tutorial()
