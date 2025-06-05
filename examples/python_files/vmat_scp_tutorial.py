"""

This example demonstrates the use of portpy.photon.vmat_scp module to create VMAT plan using sequential convex programming method.
Please refer to [Dursun et al 2021](https://iopscience.iop.org/article/10.1088/1361-6560/abee58/meta) and
[Dursun et al 2023](https://iopscience.iop.org/article/10.1088/1361-6560/ace09e/meta) for more details about the scp based algorithm

"""

import os
import portpy.photon as pp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def vmat_scp_tutorial():
    """
     1) Accessing the portpy data (DataExplorer class)

     To start using this resource, users are required to download the latest version of the dataset, which can be found at (https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit). Then, the dataset can be accessed as demonstrated below.

    """

    # specify the patient data location.
    data_dir = r'../../data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Phantom_Patient_1'

    # display the data of the patient in console or browser.
    data.display_patient_metadata()
    # display in browser rather than console. Set in_browser to True
    # data.display_patient_metadata(in_browser=True)

    # Load ct and structure set for the above patient using CT and Structures class
    ct = pp.CT(data)
    structs = pp.Structures(data)

    # Select beam ids based upon target location
    beam_ids = np.arange(0, 37)
    beams = pp.Beams(data, beam_ids=beam_ids)

    # load clinical criteria from the config files for which plan to be optimized
    protocol_name = 'Lung_2Gy_30Fx'
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    # Loading hyper-parameter values for optimization problem
    protocol_name = 'Lung_2Gy_30Fx_vmat'
    vmat_opt_params = data.load_config_opt_params(protocol_name=protocol_name)

    # # Creating optimization structures (i.e., Rinds, PTV-GTV)
    structs.create_opt_structures(opt_params=vmat_opt_params,
                                  clinical_criteria=clinical_criteria)

    # Loading influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # scale influence matrix to get correct MU/deg for TPS import
    inf_matrix_scale_factor = vmat_opt_params['opt_parameters'].get('inf_matrix_scale_factor', 1)
    inf_matrix.A = inf_matrix.A * np.float32(inf_matrix_scale_factor)

    """
     2) Creating Arc (Arcs class)

    """
    # Assign discrete beam/control_point_ids to arcs and create arcs dictionary.
    # Below is an example of creating 2 arcs. Users can create single or multiple arcs.
    arcs_dict = {'arcs': [{'arc_id': "01", "beam_ids": beam_ids[0:int(len(beam_ids) / 2)]},
                          {'arc_id': "02", "beam_ids": beam_ids[int(len(beam_ids) / 2):]}]}
    # Create arcs object using arcs dictionary and influence matrix
    arcs = pp.Arcs(arcs_dict=arcs_dict, inf_matrix=inf_matrix)

    # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria, arcs=arcs)

    """
     3) Optimize VMAT plan using sequential convex programming

    """
    # If column generation flag is turned in opt_params. Generate initial feasible leaf positions for SCP based optimization
    if vmat_opt_params['opt_parameters']['initial_leaf_pos'].lower() == 'cg':
        vmat_opt = pp.VmatOptimizationColGen(my_plan=my_plan,
                                              opt_params=vmat_opt_params)

        sol_col_gen = vmat_opt.run_col_gen_algo(solver='MOSEK', verbose=True, accept_unknown=True)
        dose_1d = inf_matrix.A @ sol_col_gen['optimal_intensity'] * my_plan.get_num_of_fractions()
        # # plot dvh for the above structures
        fig, ax = plt.subplots(figsize=(12, 8))
        struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'RIND_0', 'RIND_1', 'LUNGS_NOT_GTV', 'RECT_WALL', 'BLAD_WALL',
                        'URETHRA']
        ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose_1d,
                                       struct_names=struct_names, ax=ax)
        ax.set_title('Initial Col gen dvh')
        plt.show(block=False)
        plt.close('all')
        # pp.save_optimal_sol(sol=sol_col_gen, sol_name='sol_col_gen', path=r'C:\Temp')


    # Initialize Optimization
    vmat_opt = pp.VmatScpOptimization(my_plan=my_plan,
                                      opt_params=vmat_opt_params)
    # Run Sequential convex algorithm for optimising the plan.
    # The final result will be stored in sol and convergence will store the convergence history (i.e., results of each iteration)
    convergence = vmat_opt.run_sequential_cvx_algo(solver='MOSEK', verbose=True)
    sol = convergence[vmat_opt.best_iteration]
    sol['inf_matrix'] = inf_matrix

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
    pp.Visualization.plot_dvh(my_plan, sol=sol, struct_names=struct_names, style='solid', ax=ax)
    plt.show()
    print('Done')

    # Evaluate clinical criteria
    pp.Evaluation.display_clinical_criteria(my_plan=my_plan, sol=sol)

    # saving and loading the plan for future use (utils)
    # # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol, sol_name='sol_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # # my_plan = pp.load_plan(plan_name='my_plan_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # # sol = pp.load_optimal_sol(sol_name='sol_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))

    # write plan to dicom file
    # create dicom RT Plan file to be imported in TPS
    # We suggest you to go through notebook [VMAT_TPS_import.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/eclipse_integration.ipynb)
    # to learn about how to address the dependencies between optimization and final dose calculation before importing the plan into TPS.
    out_rt_plan_file = r'C:\Temp\Lung_Patient_3\rt_plan_portpy_vmat.dcm'  # change this file directory based upon your needs
    in_rt_plan_file = r'C:\Temp\Lung_Patient_3\rt_plan_echo_vmat.dcm'  # change this directory as per your
    pp.write_rt_plan_vmat(my_plan=my_plan, in_rt_plan_file=in_rt_plan_file, out_rt_plan_file=out_rt_plan_file)


if __name__ == "__main__":
    vmat_scp_tutorial()
