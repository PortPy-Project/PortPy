"""
PortPy provides pre-computed data with pre-defined resolutions. This example demonstrates the following down-sampling processes:
 0- Original Plan
 1- Down-sampling beamlets
 2- Calculating the plan quality cost associated with beamlet down-sampling
 3- Down-sampling the voxels
 4- Calculating the plan quality cost associated with voxel down-sampling
"""

import portpy.photon as pp
import matplotlib.pyplot as plt


def down_sampling():
    """

    1) Creating a plan using the original data resolution
    """
    # specify the patient data location.
    data_dir = r'../data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)
    # Pick a patient
    data.patient_id = 'Lung_Patient_2'
    # Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    # Pick a protocol
    protocol_name = 'Lung_2Gy_30Fx'
    # Load clinical criteria for a specified protocol
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    # Load hyper-parameter values for optimization problem for a specified protocol
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    # Create optimization structures (i.e., Rinds)
    structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)
    # Load influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # Create a plan using ct, structures, beams and influence matrix, and clinical criteria
    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria)

    # Create cvxpy problem using the clinical criteria and optimization parameters
    opt = pp.Optimization(my_plan, opt_params=opt_params, clinical_criteria=clinical_criteria)
    opt.create_cvxpy_problem()
    # Solve the cvxpy problem using Mosek
    sol_orig = opt.solve(solver='MOSEK', verbose=False)

    """
     2) Creating a plan using down-sampled beamlets 
     
    """
    # Note: PortPy only allows down-sampling beamlets as a factor of original finest beamlet resolution
    #   e.g if the finest beamlet resolution is 2.5mm (often the case) then down sampled beamlet can be 5, 7.5, 10mm
    # Down sample beamlets by a factor of 4
    beamlet_down_sample_factor = 4
    # Calculate the new beamlet resolution
    print('Finest beamlet width is {} mm and height is {} mm'.format(beams.get_finest_beamlet_width(),
                                                                     beams.get_finest_beamlet_height()))

    print('Data beamlet width is {} mm and height is {} mm'.format(beams.get_beamlet_width(),
                                                                   beams.get_beamlet_height()))
    new_beamlet_width_mm = my_plan.beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = my_plan.beams.get_finest_beamlet_height() * beamlet_down_sample_factor
    # Calculate the new beamlet resolution
    inf_matrix_db = inf_matrix.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                  beamlet_height_mm=new_beamlet_height_mm)

    # running optimization using downsampled beamlets
    # create cvxpy problem with max and mean dose clinical criteria and the above objective functions
    opt = pp.Optimization(my_plan, opt_params=opt_params,
                          inf_matrix=inf_matrix_db)
    opt.create_cvxpy_problem()
    sol_db = opt.solve(solver='MOSEK', verbose=False)

    """
    3) Cost of Down-sampling beamlets
    
    """

    # To know the cost of down sampling beamlets, lets compare the dvh of down sampled beamlets with original
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_orig, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_db, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title('Cost of Down-Sampling Beamlets  - Original .. Down-Sampled beamlets')
    plt.show()

    """
     4) Creating plan using down-sampled voxels 

    """
    # Note: PortPy only allows down-sampling voxels as a factor of ct voxel resolutions
    # PortPy can down-sample optimization voxels as factor of ct voxels.
    # Down sample voxels by a factor of 5 in x, y and 1 in z direction
    voxel_down_sample_factors = [5, 5, 1]

    # Calculate the new voxel resolution
    print('CT voxel resolution in xyz is {} mm'.format(ct.get_ct_res_xyz_mm()))
    print('Data optimization voxel resolution in xyz is {} mm'.format(
        structs.opt_voxels_dict['dose_voxel_resolution_xyz_mm']))
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    print('Down Sampled optimization voxel resolution in xyz is {} mm'.format(opt_vox_xyz_res_mm))
    inf_matrix_dv = inf_matrix.create_down_sample(opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # running optimization using down sampled voxels
    opt = pp.Optimization(my_plan, opt_params=opt_params, inf_matrix=inf_matrix_dv)
    opt.create_cvxpy_problem()
    sol_dv = opt.solve(solver='MOSEK', verbose=False)

    """
    5) Cost of voxels down-sampling
    
    Down-sampling beamlets and voxels impacts plan quality in different ways. Down-sampling beamlets can be equated to 
    merging some neighboring beamlets, effectively requiring some adjacent beamlets to share the same intensity. 
    This can lead to a compromise in plan quality.

    On the other hand, down-sampling voxels is analogous to merging some neighboring voxels. 
    This leads to less accurate dose calculations during optimization and consequently results in a discrepancy 
    between the optimized dose calculated using the original matrix and the down-sampled matrix. 

    """
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    # To calculate the accurate dose for the down-sampled solution, we first replace the down-sampled influence matrix
    # with the original influence matrix
    sol_dv_new = inf_matrix_dv.sol_change_inf_matrix(sol_dv, inf_matrix=sol_orig['inf_matrix'])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_orig, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dv_new, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title('Cost of Down-Sampling Voxels  - Original .. Down-Sampled Voxels')
    plt.show()

    # Calculate the discrepancy
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dv_new, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dv, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title(
        'Discrepancy due to Down-Sampling Voxels  \n - Down sampled with original influence matrix \n .. Down sampled without original influence matrix')
    plt.show()


if __name__ == "__main__":
    down_sampling()
