"""
    This example demonstrates the use of portpy_photon to create down sampled influence matrix and
    optimize the plan by selecting optimal beams using beam angle optimization
    1- Creating a plan using the manually selected beams
    2- Obtaining the benchmark global optimal solution
    3- Comparing the plans obtained with manually/optimally selected beams

"""
import portpy.photon as pp
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def beam_orientation_optimization():
    """
    1) Creating a plan using the manually selected beams

     **Note**: If you're developing your own beam orientation optimization algorithm, it's important to compare your results
     with manually selected beams using full-resolution data.
     However, when you're benchmarking your algorithm against the global optimal solution provided by MIP, it's important to
     use downsampled data in both MIP and your algorithm for a fair comparison.

    In this example, as we're comparing the results of MIP with manually selected beams, we apply downsampling to both scenarios.

    """
    # specify the patient data location.
    data_dir = r'../../data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)
    # Pick a patient
    data.patient_id = 'Lung_Phantom_Patient_1'
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

    # create a down-sampled influence matrix
    voxel_down_sample_factors = [7, 7, 2]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    beamlet_down_sample_factor = 4
    new_beamlet_width_mm = beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = beams.get_finest_beamlet_height() * beamlet_down_sample_factor

    inf_matrix_dbv = inf_matrix.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                   beamlet_height_mm=new_beamlet_height_mm,
                                                   opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # Create a plan using ct, structures, beams and influence matrix, and clinical criteria
    plan_planner = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix_dbv,
                           clinical_criteria=clinical_criteria)

    # Create cvxpy problem using the clinical criteria and optimization parameters
    opt = pp.Optimization(plan_planner, opt_params=opt_params, clinical_criteria=clinical_criteria)
    opt.create_cvxpy_problem()
    # Solve the cvxpy problem using Mosek
    sol_planner = opt.solve(solver='MOSEK', verbose=False)

    """
    2) Obtaining the benchmark global optimal solution
    
    **Note**: We have chosen 7 beams from a relatively small pool of 
    24 potential candidates due to computational constraints associated with Mixed Integer Programming
        
    """
    # roughly select beam_ids around planner beams
    beam_ids = list(np.arange(0, 72, 3))
    beams_boo = pp.Beams(data, beam_ids=beam_ids)

    # load the influence matrix
    inf_matrix_boo = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams_boo)
    # create a down-sampled influence matrix
    voxel_down_sample_factors = [7, 7, 2]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    beamlet_down_sample_factor = 4
    new_beamlet_width_mm = beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = beams.get_finest_beamlet_height() * beamlet_down_sample_factor

    inf_matrix_boo_dbv = inf_matrix_boo.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                           beamlet_height_mm=new_beamlet_height_mm,
                                                           opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)
    # create a plan
    plan_boo = pp.Plan(ct=ct, structs=structs, beams=beams_boo, inf_matrix=inf_matrix_boo_dbv,
                       clinical_criteria=clinical_criteria)
    # create a cvxpy optimization instance
    opt = pp.Optimization(plan_boo, opt_params=opt_params)
    opt.create_cvxpy_problem()

    # add binary variables and constraints for beam orientation optimization
    b = cp.Variable(len(inf_matrix_boo_dbv.beamlets_dict), boolean=True)
    num_beams = 7
    x = opt.vars['x']

    opt.constraints += [cp.sum(b) <= num_beams]
    for i in range(len(inf_matrix_boo_dbv.beamlets_dict)):
        start_beamlet = inf_matrix_boo_dbv.beamlets_dict[i]['start_beamlet_idx']
        end_beamlet = inf_matrix_boo_dbv.beamlets_dict[i]['end_beamlet_idx']
        M = 50  # upper bound on the beamlet intensity
        opt.constraints += [x[start_beamlet:end_beamlet] <= b[i] * M]
    # solve the problem
    sol_boo = opt.solve(solver='MOSEK', verbose=False)

    """
    3) Comparing the plans obtained with manually/optimally selected beams 
    
    """
    # Identifying bao and planner gantry angles
    boo_gantry_angles = (np.asarray(plan_boo.beams.beams_dict['gantry_angle']) + 1) * (b.value > 0)
    boo_gantry_angles = boo_gantry_angles[boo_gantry_angles > 0] - 1  # add and subtract 1 to check for 0 degree angle
    print('BOO gantry angles: {}'.format(boo_gantry_angles))
    planner_gantry_angles = plan_planner.beams.beams_dict['gantry_angle']
    print('Planner gantry angles: {}'.format(planner_gantry_angles))

    # plot dvh dvh for both the cases
    fig, ax = plt.subplots(figsize=(12, 8))
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNG_L', 'LUNG_R', 'RIND_0', 'RIND_1']
    ax = pp.Visualization.plot_dvh(plan_boo, sol=sol_boo, struct_names=struct_names, style='solid', ax=ax)
    ax = pp.Visualization.plot_dvh(plan_planner, sol=sol_planner, struct_names=struct_names, style='dotted', ax=ax)
    ax.set_title('- BOO  .. Planner')
    plt.show()


if __name__ == "__main__":
    beam_orientation_optimization()
