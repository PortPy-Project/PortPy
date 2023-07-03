"""

This example demonstrates the following main functionalities of portpy_photon:
    1- Accessing the curated portpy data
        (DataExplorer class)

    2- Creating a simple IMRT plan
        (Plan class, Optimization class)

    3- Visualising the plan (e.g., dose distribution, fluence, DVH)
       (Visualization class)

    4- Evaluating the plan (e.g., max/mean/DVH points, established clinical metrics)
        (Evaluation class)

    5- Saving the plan and solution for future uses
        (Utils)

    6- Advanced plan visualization through integration with a popular 3D-Slicer package
        (Visualization class)

"""
import portpy.photon as pp
import os


def ex_1_introduction():
    """
     1) accessing the portpy data (DataExplorer class)
     Note: you first need to download the patient database from the link provided in the GitHub page.

    """

    # specify the patient data location.
    data_dir = r'../data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)
    # display the existing patients in console or browser.
    data.display_list_of_patients()

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Phantom_Patient_1'
    # display the data of the patient in console or browser.
    data.display_patient_metadata()
    # display in browser rather than console. Set in_browser to True
    # data.display_patient_metadata(in_browser=True)

    # Load ct and structure set for the above patient using CT and Structures class
    ct = pp.CT(data)
    structs = pp.Structures(data)

    # If the list of beams are not provided, it uses the beams selected manually
    # by a human expert planner for the patient (manually selected beams are stored in portpy data).
    # Create beams for the planner beams by default
    # for the customized beams, you can pass the argument beam_ids
    # e.g. beams = pp.Beams(data, beam_ids=[0,10,20,30,40,50,60])
    beams = pp.Beams(data)

    # create rinds based upon rind definition in optimization params
    protocol_name = 'Lung_2Gy_30Fx'
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    structs.create_opt_structures(opt_params)

    # load influence matrix based upon beams and structure set
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # load clinical criteria from the config files for which plan to be optimized
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    """
    2) creating a simple IMRT plan using CVXPy (Plan class, Optimization class)
    Note: you can call different opensource / commercial optimization engines from CVXPy.
      For commercial engines (e.g., Mosek, Gorubi, CPLEX), you first need to obtain an appropriate license.
      Most commercial optimization engines give free academic license.

    Create my_plan object which would store all the data needed for optimization
      (e.g., influence matrix, structures and their voxels, beams and their beamlets).
    
    """
    # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan = pp.Plan(ct, structs, beams, inf_matrix, clinical_criteria)

    # create cvxpy problem using the clinical criteria and optimization parameters
    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    sol = opt.solve(solver='MOSEK', verbose=True)

    """ 
    3) visualizing the plan (Visualization class)
    
    """
    # plot fluence 3d and 2d for the 1st beam
    pp.Visualization.plot_fluence_3d(sol=sol, beam_id=my_plan.beams.get_all_beam_ids()[0])

    pp.Visualization.plot_fluence_2d(sol=sol, beam_id=my_plan.beams.get_all_beam_ids()[0])

    # plot dvh for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    # plot dvh for the above structures
    pp.Visualization.plot_dvh(my_plan, sol=sol, struct_names=struct_names, title=data.patient_id)

    # plot 2d axial slice for the given solution and display the structures contours on the slice
    pp.Visualization.plot_2d_slice(my_plan=my_plan, sol=sol, slice_num=60, struct_names=['PTV'])

    """ 
    4) evaluating the plan (Evaluation class) 
    
    """

    # visualize plan metrics based upon clinical criteria
    pp.Evaluation.display_clinical_criteria(my_plan, sol=sol)

    """ 
    5) saving and loading the plan for future use (utils) 
    
    """
    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan_phantom.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol, sol_name='sol_phantom.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # my_plan = pp.load_plan(plan_name='my_plan_phantom.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # sol = pp.load_optimal_sol(sol_name='sol_phantom.pkl', path=os.path.join(r'C:\temp', data.patient_id))

    """
    6) Advanced visualization with 3D-Slicer (Visualization class) 
    
    """
    # Note: This requires downloading and installing 3d slicer
    #   3D slicer integration is seamless with Jupiter Notebook (see ex_7_Slicer).
    # Without Jupiter Notebook, we first need to save our data (ct, 3D dose, structures)
    #   as images in nrrd format on disk and then lunch Slicer by providing data as input argument
    pp.save_nrrd(my_plan, sol=sol, data_dir=os.path.join(r'C:\temp', data.patient_id))
    pp.Visualization.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 5.2.1\Slicer.exe',
                                    data_dir=os.path.join(r'C:\temp', data.patient_id))
    print('Done!')


if __name__ == "__main__":
    ex_1_introduction()
