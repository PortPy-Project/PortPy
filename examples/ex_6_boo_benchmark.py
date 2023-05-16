"""
    This example demonstrates the use of portpy_photon to create down sampled influence matrix and
    optimize the plan by selecting optimal beams using beam angle optimization
    1- Down sample influence matrix and create a plan using planner beams
    2- Create a plan using beam angle optimization (boo)
    3- Evaluate and compare the plans

"""
import portpy.photon as pp
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def ex_6_boo_benchmark():
    """
    1) Create down sampled influence matrix and generate a plan using planner beams

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

    plan_planner = pp.Plan(ct, structs, beams, inf_matrix_dbv, clinical_criteria)

    opt = pp.Optimization(plan_planner, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol_planner = opt.solve(solver='MOSEK', verbose='True')

    """
    2) Select optimal beams using beam angle optimization and create plan using it
    
    """
    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    # creating plan and select among the beams which are 30 degrees apart
    # beam_ids = [0, 37, 42, 48, 54, 60, 66]
    beam_ids = [0, 6, 12, 18, 24, 30, 37, 42, 48, 54, 60, 66]
    beams_boo = pp.Beams(data, beam_ids=beam_ids)

    # create a influence matrix down sampled beamlets of width and height 5mm
    # load influence matrix based upon beams and structure set
    inf_matrix_boo = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_boo)

    # create a influence matrix down sampled beamlets of width and height 5mm
    voxel_down_sample_factors = [7, 7, 2]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    beamlet_down_sample_factor = 4
    new_beamlet_width_mm = beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = beams.get_finest_beamlet_height() * beamlet_down_sample_factor

    inf_matrix_boo_dbv = inf_matrix_boo.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                           beamlet_height_mm=new_beamlet_height_mm,
                                                           opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    plan_boo = pp.Plan(ct, structs, beams_boo, inf_matrix_boo_dbv, clinical_criteria)

    opt = pp.Optimization(plan_boo, opt_params=opt_params)
    opt.create_cvxpy_problem()

    st = inf_matrix_boo_dbv

    # binary variable for selecting beams
    b = cp.Variable(len(st.beamlets_dict), boolean=True)
    num_beams = 7
    x = opt.vars['x']

    opt.constraints += [cp.sum(b) <= num_beams]
    for i in range(len(st.beamlets_dict)):
        start_beamlet = st.beamlets_dict[i]['start_beamlet_idx']
        end_beamlet = st.beamlets_dict[i]['end_beamlet_idx']
        M = 50  # upper bound on the beamlet intensity
        opt.constraints += [x[start_beamlet:end_beamlet] <= b[i] * M]
    sol_boo = opt.solve(solver='MOSEK', verbose='True')

    # Comment/Uncomment these lines to save & load plan and optimal solutions
    # plan_boo = pp.load_plan(plan_name='plan_boo', path=r'C:\temp')
    # sol_boo = pp.load_optimal_sol('sol_boo_w_sm_10', path=r'C:\temp')
    # plan_planner = pp.load_plan(plan_name='plan_planner', path=r'C:\temp')
    # sol_planner = pp.load_optimal_sol('sol_planner', path=r'C:\temp')
    pp.save_plan(plan_boo, plan_name='plan_boo', path=r'C:\temp')
    pp.save_optimal_sol(sol_boo, sol_name='sol_boo', path=r'C:\temp')

    """
    3) Evaluating the plans
    
    """
    # Identifying bao and planner gantry angles
    boo_gantry_angles = (np.asarray(plan_boo.beams.beams_dict['gantry_angle']) + 1) * (sol_boo['optimal_beams'] > 0)
    boo_gantry_angles = boo_gantry_angles[boo_gantry_angles > 0] - 1  # add and subtract 1 to check for 0 degree angle
    print('BOO gantry angles: {}'.format(boo_gantry_angles))
    planner_gantry_angles = plan_planner.beams.beams_dict['gantry_angle']
    print('Planner gantry angles: {}'.format(planner_gantry_angles))

    # plot dvh dvh for both the cases
    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    ax = pp.Visualization.plot_dvh(plan_boo, sol=sol_boo, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(plan_planner, sol=sol_planner, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title('- BOO  .. Planner')
    plt.show()


if __name__ == "__main__":
    ex_6_boo_benchmark()
