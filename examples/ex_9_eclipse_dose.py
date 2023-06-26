"""

This example demonstrates the following main functionalities of portpy_photon:
    1- Accessing the curated portpy data
        (DataExplorer class)

    2- Creating a simple IMRT plan
        (Plan class, Optimization class)

    3- Saving the eclipse fluence and push it to eclipse (Utils class)

    4- Export RT DICOM dose and convert it into portpy format (Utils class)

    5- Visualize the dose discrepancy (Visualization Class)


"""
import portpy.photon as pp
import os
import matplotlib.pyplot as plt


def ex_9_eclipse_dose():
    """
     1) Accessing the portpy data (DataExplorer class)
     Note: you first need to download the patient database from the link provided in the GitHub page.

    """

    # specify the patient data location.
    data_dir = r'../data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)
    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Patient_2'

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

    sol = opt.solve(solver='MOSEK', verbose=True)

    """ 
    3) saving the eclipse fluence and push it to eclipse 

    """

    pp.get_eclipse_fluence(sol, path=os.path.join(r'C:\temp', data.patient_id))
    # beam_ids=['Field 1', 'Field 2', 'Field 3', 'Field 4', 'Field 5', 'Field 6', 'Field 7'])

    # To push it to eclipse. Load RT plan in eclipse and right click each field and Select "Import Optimal Fluence" and
    # select the fluence from the path directory and import the fluence

    """
    4) Export RT DICOM dose and convert it into portpy format 

    """
    dir_name = r'C:\Lung\Test\Outputs\NSCLC_LUNG1-002$LUNG1-002\PortPy_5\Dose'  # modify the directory name to the directory you exported dose
    ecl_dose_3d = pp.convert_dose_rt_dicom_to_portpy(my_plan=my_plan, dir_name=dir_name)
    ecl_dose_1d = inf_matrix.dose_3d_to_1d(dose_3d=ecl_dose_3d)

    """
    5) Visualize the dose discrepancy between optimization and eclipse dose
    
    """
    # Visualize the DVH discrepancy
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=ecl_dose_1d, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title('- optimization .. Eclipse')
    plt.show()
    print('Done!')


if __name__ == "__main__":
    ex_9_eclipse_dose()
