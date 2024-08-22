"""

This example demonstrates the following main functionalities of portpy_photon:
    1- Accessing the curated portpy data
        (DataExplorer class)

    2- Creating a simple proton plan
        (Plan class, Optimization class)

    3- Visualising the plan (e.g., dose distribution, DVH)
       (Visualization class)

    4- Evaluating the plan (e.g., max/mean/DVH points, established clinical metrics)
        (Evaluation class)

    5- Saving the plan and solution for future uses
        (Utils)

    6- Advanced plan visualization through integration with a popular 3D-Slicer package
        (Visualization class)

"""
import os
import portpy.proton as pp
from portpy.proton.utils.create_ct_dose_voxel_map import create_ct_dose_voxel_map
import matplotlib.pyplot as plt

def proton_tutorial():
    """
     1) Accessing the portpy data (DataExplorer class)

     PortPy provides researchers with a comprehensive benchmark patient dataset derived from an FDA-approved Eclipse commercial treatment planning system via its API.
     This dataset includes all the necessary components for optimizing various machine settings such as beam angles, aperture shapes, and leaf movements.
     In addition to the CT images and delineated contours, the dataset includes:
     1. **Dose Influence Matrix:** The dose contribution of each beamlet to each voxel,
     2. **Beamlets/Voxels Details:** Detailed information about the position and size of beamlets/voxels,
     3. **Expert-Selected Benchmark Beams:** An expert clinical physicist has carefully selected benchmark beams,
     providing reference beams for comparison and benchmarking,
     4. **Benchmark IMRT Plan:** A benchmark IMRT plan generated using MSK in-house automated treatment planning
     system called [ECHO](https://youtu.be/895M6j5KjPs). This plan serves as a benchmark for evaluating new treatment planning algorithms.

     To start using this resource, users are required to download the latest version of the dataset, which can be found at (https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit). Then, the dataset can be accessed as demonstrated below.

    """

    # specify the patient data location.
    data_dir = r'F:\HaiTest\MAAS-ProtonInfluenceMatrix\Software\results'
    # data_dir = r'F:\HaiTest\MAAS-ProtonInfluenceMatrix'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)
    # display the existing patients in console or browser.
    # data.display_list_of_patients()

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'ProtonTestSM1'
    # display the data of the patient in console or browser.
    # data.display_patient_metadata()
    # display in browser rather than console. Set in_browser to True
    # data.display_patient_metadata(in_browser=True)

    # Load ct and structure set for the above patient using CT and Structures class
    # ct = pp.CT(data)

    # If the list of beams are not provided, it uses the beams selected manually
    # by a human expert planner for the patient (manually selected beams are stored in portpy data).
    # Create beams for the planner beams by default
    # for the customized beams, you can pass the argument beam_ids
    # e.g. beams = pp.Beams(data, beam_ids=[0,10,20,30,40,50,60])
    beams = pp.Beams(data)

    # create a dictionary called map structures which map dicom structure name to standard structure name used in PortPy clinical criteria.
    # Eg. 'PTVSmall' is mapped to 'PTV' structure
    # User needs to change it based on their patient specific structure names
    map_structures = {'PTVSmall': 'PTV',
                      'EL_Skin': 'SKIN',
                      'Liver': ' LIVER',
                      'Stomach38': 'STOMACH',
                      'Lungs_NOT_GTV38': 'LUNGS_NOT_GTV',
                      'Lung_R38': 'LUNG_R',
                      'Lung_L38': 'LUNG_L',
                      'ITV38': 'ITV',
                      'Heart38': 'HEART',
                      'GTV38': 'GTV',
                      'Esophagus38': 'ESOPHAGUS',
                      'CTV38': 'CTV',
                      'Cord38': 'CORD',
                      'Patient Surfac38': 'BODY'}
    # get num points from influence matrix rows
    # num_points = beams.beams_dict['influenceMatrixSparse'][0].shape[0]
    # Create ct dose voxel map. It modify raw data to create data in PortPy format
    # create_ct_dose_voxel_map(os.path.join(data.data_dir, data.patient_id), num_points=num_points)
    structs = pp.Structures(data)

    dicom_structure_names = map_structures.keys()
    for ind, s in enumerate(structs.get_structures()):
        if s in dicom_structure_names:
            structs.structures_dict['name'][ind] = map_structures[s]

    # In order to create an IMRT plan, we first need to specify a protocol which includes the disease site,
    # the prescribed dose for the PTV, the number of fractions, and the radiation dose thresholds for OARs.
    # These information are stored in .json files which can be found in a directory named "config_files".
    # An example of such a file is 'Lung_2Gy_30Fx.json'. Here's how you can load these files:
    protocol_name = 'Lung_2Gy_30Fx'
    # load clinical criteria from the config files for which plan to be optimized
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    # Creating optimization structures (i.e., Rinds)
    structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)
    # Loading influence matrix
    inf_matrix = pp.InfluenceMatrix(structs=structs, beams=beams, is_bev=True)

    """

    2) creating a simple IMRT plan using CVXPy (Plan class, Optimization class)
    Note: you can call different opensource / commercial optimization engines from CVXPy.
      For commercial engines (e.g., Mosek, Gorubi, CPLEX), you first need to obtain an appropriate license.
      Most commercial optimization engines give free academic license.

    Create my_plan object which would store all the data needed for optimization
      (e.g., influence matrix, structures and their voxels, beams and their beamlets).

    """
    # create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan = pp.Plan(structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)

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

    # plot dvh for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNG_L', 'LUNG_R']

    # plot dvh for the above structures
    pp.Visualization.plot_dvh(my_plan, sol=sol, struct_names=struct_names, title=data.patient_id)

    plt.show()
    # plot 2d axial slice for the given solution and display the structures contours on the slice
    pp.Visualization.plot_2d_slice(my_plan=my_plan, sol=sol, slice_num=60, struct_names=['PTV'])

    """ 
    4) evaluating the plan (Evaluation class) 
    The Evaluation class offers a set of methods for quantifying the optimized plan. 
    If you need to compute individual dose volume metrics, you can use methods such as *get_dose* or *get_volume*. 
    Furthermore, the class also facilitates the assessment of the plan based on a collection of metrics, 
    such as mean, max, and dose-volume histogram (DVH), as specified in the clinical protocol. This capability is demonstrated below
    """

    # visualize plan metrics based upon clinical criteria
    pp.Evaluation.display_clinical_criteria(my_plan, sol=sol)

    """ 
    5) saving and loading the plan for future use (utils) 

    """
    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan_proton.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol, sol_name='sol_proton.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # my_plan = pp.load_plan(plan_name='my_plan_phantom.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # sol = pp.load_optimal_sol(sol_name='sol_phantom.pkl', path=os.path.join(r'C:\temp', data.patient_id))

    print('Done!')


if __name__ == "__main__":
    proton_tutorial()
