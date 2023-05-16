"""
    This example demonstrates the use of portpy_photon to create down sampled influence matrix and
    optimize it with exact dvh constraints for benchmarking
    1- Down sample influence matrix and create a plan without dvh constraint
    2- Add exact DVH constraint and create a plan
    3- Evaluate and compare the plans
"""
import portpy.photon as pp
import matplotlib.pyplot as plt


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

    # create a influence matrix down sampled beamlets of width and height 5mm
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
    # optimize with downscaled influence matrix and the dvh constraints created below
    eso_dvh = clinical_criteria.create_criterion(criterion='dose_volume_V',
                                                 parameters={'structure_name': 'ESOPHAGUS', 'dose_gy': 60},
                                                 constraints={'limit_volume_perc': 17})
    opt.add_dvh(dvh_constraint=eso_dvh)
    sol_dvh = opt.solve(solver='MOSEK', verbose='True')

    # Comment/Uncomment these lines to save & load plan and optimal solutions
    # my_plan.save_plan(path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_no_dvh, sol_name='sol_no_dvh', path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_dvh, sol_name='sol_dvh', path=r'C:\temp')
    # my_plan = Plan.load_plan(path=r'C:\temp')
    # sol_no_dvh = Plan.load_optimal_sol('sol_no_dvh', path=r'C:\temp')
    # sol_dvh = Plan.load_optimal_sol('sol_dvh', path=r'C:\temp')

    # plot dvh dvh for both the cases
    """
    3) Evaluate the plans with and without DVH constraint
    
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_no_dvh, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(my_plan, sol=sol_dvh, struct_names=struct_names, style='dotted',
                                   show_criteria=eso_dvh, ax=ax)
    ax.set_title('- Without DVH  .. With DVH')
    plt.show()

    # visualize 2d dose_1d for both the cases
    pp.Visualization.plot_2d_slice(my_plan, sol=sol_no_dvh, struct_names=struct_names)
    pp.Visualization.plot_2d_slice(my_plan, sol=sol_dvh, struct_names=struct_names)

    print('Done!')


if __name__ == "__main__":
    ex_5_dvh_benchmark()
