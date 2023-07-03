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


def ex_2_down_sampling():
    """

    0) Creating a plan using the original data resolution
    """
    # Create my_plan object for the planner beams.
    data_dir = r'../data'
    # display the existing patients in console or browser.
    data = pp.DataExplorer(data_dir=data_dir)
    data.patient_id = 'Lung_Phantom_Patient_1'

    # Load ct, structure and beams as an object
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)

    # create extra optimization structures based upon structure definition in optimization params
    protocol_name = 'Lung_2Gy_30Fx'
    opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    structs.create_opt_structures(opt_params)

    # load influence matrix based upon beams and structure set
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # load clinical criteria from the config files for which plan to be optimized
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    my_plan = pp.Plan(ct, structs, beams, inf_matrix, clinical_criteria)

    # ### Run Optimization
    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol_orig = opt.solve(solver='MOSEK', verbose=True)

    """
     1) Creating a plan using down-sampled beamlets 
     
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
    2) Cost of Down-sampling beamlets
    
    """

    # To know the cost of down sampling beamlets, lets compare the dvh of down sampled beamlets with original
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_orig, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_db, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title('Cost of Down-Sampling Beamlets  - Original .. Down-Sampled beamlets')
    plt.show()

    """
     3) Creating plan using down-sampled voxels 

    """
    # Note: PortPy only allows down-sampling voxels as a factor of ct voxel resolutions resolution
    # PortPy can down-sample optimization voxels as factor of ct voxels.
    # Down sample voxels by a factor of 7 in x, y and 2 in z direction
    voxel_down_sample_factors = [7, 7, 2]

    # Calculate the new voxel resolution
    print('CT voxel resolution in xyz is {} mm'.format(ct.get_ct_res_xyz_mm()))
    print('Data optimization voxel resolution in xyz is {} mm'.format(structs.opt_voxels_dict['dose_voxel_resolution_xyz_mm']))
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    inf_matrix_dv = inf_matrix.create_down_sample(opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # running optimization using down sampled voxels
    # create cvxpy problem with max and mean dose clinical criteria and the above objective functions
    opt = pp.Optimization(my_plan, opt_params=opt_params, inf_matrix=inf_matrix_dv)
    opt.create_cvxpy_problem()
    sol_dv = opt.solve(solver='MOSEK', verbose=False)

    """
    4) Cost and discrepancy of Down-sampling voxels 

    """
    # Similarly to analyze the cost of down sampling voxels, lets compare the dvh of down sampled voxels with original
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    sol_dv_new = inf_matrix_dv.sol_change_inf_matrix(sol_dv, inf_matrix=sol_orig['inf_matrix'])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_orig, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dv_new, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title('Cost of Down-Sampling Voxels  - Original .. Down-Sampled Voxels')
    plt.show()

    # To get the discrepancy due to down sampling voxels
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dv_new, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dv, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title(
        'Discrepancy due to Down-Sampling Voxels  \n - Down sampled with original influence matrix \n .. Down sampled without original influence matrix')
    plt.show()


if __name__ == "__main__":
    ex_2_down_sampling()
