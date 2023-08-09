"""
    This example demonstrates the use of portpy_photon to create down sampled influence matrix and
    optimize it with exact dvh constraints for benchmarking
    1- Creating an IMRT plan without DVH constraints on down-sampled data
    2- Obtaining the benchmark global optimal solution
    3- Comparing the plans with/without DVH constraints
"""
import portpy.photon as pp
import matplotlib.pyplot as plt
import cvxpy as cp


def dvh_constraint_optimization():
    """
    1) Creating an IMRT plan without DVH constraints on down-sampled data

    **Note:** When benchmarking your DVH constraint algorithm against the globally optimal solution derived from MIP,
    it's crucial to use downsampled data in both the MIP and your algorithm. This ensures a fair comparison.
    In this example, we are only comparing the plan without DVH constraints against the benchmark MIP plan.

    """
    # Pick a patient
    data_dir = r'../../data'
    data = pp.DataExplorer(data_dir=data_dir)
    patient_id = 'Lung_Patient_7'
    data.patient_id = patient_id

    # Load ct, structure and beams objects
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

    # create down-sampled influence matrix
    voxel_down_sample_factors = [6, 6, 1]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    beamlet_down_sample_factor = 4
    new_beamlet_width_mm = beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = beams.get_finest_beamlet_height() * beamlet_down_sample_factor

    inf_matrix_dbv = inf_matrix.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                   beamlet_height_mm=new_beamlet_height_mm,
                                                   opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # create a plan

    my_plan = pp.Plan(ct, structs, beams, inf_matrix_dbv, clinical_criteria)

    # create a cvxpy optimization object
    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()
    # solve the problem
    sol_no_dvh = opt.solve(solver='MOSEK', verbose=False)

    """
    2) Obtaining the benchmark global optimal solution

    """
    # Add a dvh constraint V(20Gy) <= 10% for ESOPHAGUS as shown below
    dvh_org = 'ESOPHAGUS'
    dose_gy = 20
    limit_volume_perc = 10

    # extract data for dvh constraint
    A = inf_matrix_dbv.A  # down sample influence matrix
    x = opt.vars['x']  # optimization variable
    M = 50  # set Big M for dvh constraint
    # Get fraction of organ volume inside the dose calculation box.

    # Add binary variables and constraints for dvh constraint
    b = cp.Variable(
        len(inf_matrix_dbv.get_opt_voxels_idx('CORD')),
        boolean=True)
    opt.constraints += [A[inf_matrix_dbv.get_opt_voxels_idx(dvh_org), :] @ x <= dose_gy / my_plan.get_num_of_fractions() + b * M]
    opt.constraints += [b @ inf_matrix_dbv.get_opt_voxels_volume_cc(dvh_org) <= limit_volume_perc / 100 * sum(
        inf_matrix_dbv.get_opt_voxels_volume_cc(dvh_org))]

    sol_dvh = opt.solve(solver='MOSEK', verbose=False)

    # plot dvh dvh for both the cases
    """
    3) Comparing the plans with/without DVH constraints
    
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_no_dvh, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dvh, struct_names=struct_names, style='dotted', ax=ax)
    ax.plot(dose_gy, limit_volume_perc, marker='x', color='red', markersize=20)
    ax.set_title('- Without DVH  .. With DVH')
    plt.show()
    print('Done!')

    print('Done!')


if __name__ == "__main__":
    dvh_constraint_optimization()
