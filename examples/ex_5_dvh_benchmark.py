"""
    This example demonstrates the use of portpy_photon to create down sampled influence matrix and
    optimize it with exact dvh constraints for benchmarking
    1- Down sample influence matrix and create a plan without dvh constraint
    2- Add exact DVH constraint and create a plan
    3- Evaluate and compare the plans
"""
import portpy.photon as pp
import matplotlib.pyplot as plt
import cvxpy as cp


def ex_5_dvh_benchmark():
    """
    1) Create down sampled influence matrix and generate a plan without DVH constraint

    """
    # Create plan object
    data_dir = r'../data'
    data = pp.DataExplorer(data_dir=data_dir)
    patient_id = 'Lung_Phantom_Patient_1'
    data.patient_id = patient_id

    # Load ct, structure and beams as an object
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)

    # create rinds based upon rind definition in optimization params
    opt_params = data.load_config_opt_params(protocol_name='Lung_2Gy_30Fx')
    structs.create_opt_structures(opt_params)

    # load influence matrix based upon beams and structure set
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # create a influence matrix down sampled beamlets of width and height 10mm and down sampled voxels
    voxel_down_sample_factors = [7, 7, 2]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    beamlet_down_sample_factor = 4
    new_beamlet_width_mm = beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = beams.get_finest_beamlet_height() * beamlet_down_sample_factor

    inf_matrix_dbv = inf_matrix.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                   beamlet_height_mm=new_beamlet_height_mm,
                                                   opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # load clinical criteria from the config files for which plan to be optimized
    protocol_name = 'Lung_2Gy_30Fx'
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name)

    my_plan = pp.Plan(ct, structs, beams, inf_matrix_dbv, clinical_criteria)

    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol_no_dvh = opt.solve(solver='MOSEK', verbose='True')

    """
    2) Add exact DVH constraint and generate a plan with DVH constraint

    """
    # Add a dvh constraint V(10Gy) <= 15% for CORD as shown below
    dvh_org = 'CORD'
    dose_gy =  10
    limit_volume_perc = 15

    # extract data for dvh constraint
    A = inf_matrix_dbv.A # down sample influence matrix
    x = opt.vars['x'] # optimization variable
    M = 50 # set Big M for dvh constraint
    frac = my_plan.structures.get_fraction_of_vol_in_calc_box(dvh_org) # get fraction of dvh organ volume inside dose calculation box


    # Create binary variable for dvh constraint
    b_dvh = cp.Variable(
        len(inf_matrix_dbv.get_opt_voxels_idx('CORD')),
        boolean=True)

    # Add dvh constraint
    opt.constraints += [
        A[inf_matrix_dbv.get_opt_voxels_idx(dvh_org), :] @ x <= dose_gy / my_plan.get_num_of_fractions()
        + b_dvh * M]
    opt.constraints += [b_dvh @ inf_matrix_dbv.get_opt_voxels_volume_cc(dvh_org) <= (limit_volume_perc / frac) / 100 * sum(
        inf_matrix_dbv.get_opt_voxels_volume_cc(dvh_org))]
    sol_dvh = opt.solve(solver='MOSEK', verbose='True')

    # plot dvh dvh for both the cases
    """
    3) Visualize the plans with and without DVH constraint
    
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_no_dvh, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dvh, struct_names=struct_names, style='dotted', ax=ax)
    ax.plot(dose_gy, limit_volume_perc, marker='x', color='red', markersize=20)
    ax.set_title('- Without DVH  .. With DVH')
    plt.show()

    # visualize 2d dose_1d for both the cases
    pp.Visualization.plot_2d_slice(my_plan, sol=sol_no_dvh, struct_names=struct_names)
    pp.Visualization.plot_2d_slice(my_plan, sol=sol_dvh, struct_names=struct_names)

    print('Done!')


if __name__ == "__main__":
    ex_5_dvh_benchmark()
