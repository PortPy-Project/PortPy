"""

This example demonstrates the use of portpy.photon to create down sampled influence matrix and
create a benchmark VMAT plan

"""
import portpy.photon as pp
import numpy as np
import cvxpy as cp
import os
import time

import numpy as np

def quadratic_overdose(d, voxels, voxels_cc, D, weight):
    return weight * (1 / np.sum(voxels_cc))*np.sum(voxels_cc*(np.maximum(d[voxels] - D, 0) ** 2))

def quadratic_underdose(d, voxels, voxels_cc, D, weight):
    return weight * (1 / np.sum(voxels_cc)) * np.sum(voxels_cc*(np.maximum(D - d[voxels], 0) ** 2))

def quadratic(d, voxels,  voxels_cc, weight):
    return weight * (1 / np.sum(voxels_cc)) * np.sum(voxels_cc*(d[voxels] ** 2))

def compute_objective(my_plan, d, objectives, prescription_dose):
    total = 0.0
    for obj in objectives:
        name = obj['structure_name']
        voxels = my_plan.inf_matrix.get_opt_voxels_idx(name)
        voxels_cc = my_plan.inf_matrix.get_opt_voxels_volume_cc(name)
        if len(voxels) == 0:
            continue  # skip missing structures

        w = obj['weight']
        if obj['type'] == 'quadratic-overdose':
            dose_perc = obj['dose_perc'] / 100
            D = dose_perc * prescription_dose
            total += quadratic_overdose(d, voxels, voxels_cc, D, w)
        elif obj['type'] == 'quadratic-underdose':
            dose_perc = obj['dose_perc'] / 100
            D = dose_perc * prescription_dose
            total += quadratic_underdose(d, voxels, voxels_cc, D, w)
        elif obj['type'] == 'quadratic':
            total += quadratic(d, voxels, voxels_cc, w)
        else:
            raise ValueError(f"Unsupported objective type: {obj['type']}")
    return total

def vmat_optimization():
    """
    1. Create down sampled influence matrix and generate benchmark VMAT plan

    """
    # specify the patient data location.
    data_dir = r'../../data'
    # Use PortPy DataExplorer class to explore PortPy data
    data = pp.DataExplorer(data_dir=data_dir)
    # Pick a patient
    data.patient_id = 'Lung_Patient_6'
    # Load ct, structure set, beams for the above patient using CT, Structures, and Beams classes
    ct = pp.CT(data)
    structs = pp.Structures(data)
    # generating VMAT plan using all the control points is computationally expensive. We select only 8 equidistant beams to create the benchamrk VMAT plan.
    beam_ids = list(np.arange(0, 72, 11))
    beams = pp.Beams(data, beam_ids=beam_ids)

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
    voxel_down_sample_factors = [6, 6, 1]
    opt_vox_xyz_res_mm = [ct_res * factor for ct_res, factor in zip(ct.get_ct_res_xyz_mm(), voxel_down_sample_factors)]
    beamlet_down_sample_factor = 6
    new_beamlet_width_mm = beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = beams.get_finest_beamlet_height() * beamlet_down_sample_factor

    inf_matrix_db = inf_matrix.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                  beamlet_height_mm=new_beamlet_height_mm,
                                                  opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)

    # Create a plan
    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix_db,
                      clinical_criteria=clinical_criteria)

    # remove smoothness objective
    for i in range(len(opt_params['objective_functions'])):
        if opt_params['objective_functions'][i]['type'] == 'smoothness-quadratic':
            opt_params['objective_functions'][i]['weight'] = 0

    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()

    # create optimization problem
    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()

    # replace the quadratic objective functions with the linear ones
    obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []

    A = inf_matrix_db.A
    num_fractions = clinical_criteria.get_num_of_fractions()
    st = inf_matrix_db
    x = opt.vars['x']
    d = None

    # Construct optimization problem

    opt.obj = []  # Remove all previous objective functions

    # Generating new linear objective functions
    print('Objective Start')
    for i in range(len(obj_funcs)):
        if obj_funcs[i]['type'] == 'quadratic-overdose':
            if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                dose_gy = opt.get_num(obj_funcs[i]['dose_gy']) / clinical_criteria.get_num_of_fractions()
                dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                opt.obj += [(1 / len(st.get_opt_voxels_idx(struct))) *
                            (obj_funcs[i]['weight'] * cp.sum(dO))]
                opt.constraints += [A[st.get_opt_voxels_idx(struct), :] @ x <= dose_gy + dO]
        elif obj_funcs[i]['type'] == 'quadratic-underdose':
            if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                dose_gy = opt.get_num(obj_funcs[i]['dose_gy']) / clinical_criteria.get_num_of_fractions()
                dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                opt.obj += [(1 / len(st.get_opt_voxels_idx(struct))) *
                            (obj_funcs[i]['weight'] * cp.sum(dU))]
                opt.constraints += [A[st.get_opt_voxels_idx(struct), :] @ x >= dose_gy - dU]
        elif obj_funcs[i]['type'] == 'quadratic':
            if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                struct = obj_funcs[i]['structure_name']
                opt.obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (
                        obj_funcs[i]['weight'] * cp.sum(A[st.get_opt_voxels_idx(struct), :] @ x))]

    # data preprocessing for creating VMAT beams
    # Creating a map between leaf and beamlet indices for each control point
    beam_maps = my_plan.inf_matrix.get_bev_2d_grid(beam_id=my_plan.beams.get_all_beam_ids())
    vmat_beams = []
    for i in range(len(beam_maps)):
        beam_map = beam_maps[i]
        beam_map = beam_map[~np.all(beam_map == -1, axis=1), :]  # remove rows which are not in BEV
        # beam_map = beam_map[:, ~np.all(beam_map == -1, axis=0)]  # remove cols which are not in BEV
        num_rows, num_cols = beam_map.shape

        leaf_idx_beamlet_map = {}
        for j, row in enumerate(beam_map):
            leaf_idx_beamlet_map[j] = row[row >= 0].tolist()
        map_beamlet = {'leaf_idx_beamlet_map': leaf_idx_beamlet_map, 'num_rows': num_rows, 'num_cols': num_cols,
                       'beam_map': beam_map}
        vmat_beams.append(map_beamlet)
    total_rows = sum([b['num_rows'] for b in vmat_beams])

    # creating new variables for VMAT
    lbi = cp.Variable(total_rows, integer=True)
    rbi = cp.Variable(total_rows, integer=True)
    z = cp.Variable(my_plan.inf_matrix.A.shape[1], boolean=True)
    mu = cp.Variable(len(vmat_beams), pos=True)
    x = opt.vars['x']
    U = 2

    # Add geometric constraints (consecutive ones)
    leaf_in_prev_beam = 0
    for i, beam in enumerate(vmat_beams):
        beam_map = beam['beam_map']
        leaf_idx_beamlet_map = beam['leaf_idx_beamlet_map']

        for leaf in leaf_idx_beamlet_map:
            beamlets_in_leaf = leaf_idx_beamlet_map[leaf]
            c = np.where(np.isin(beam_map, beamlets_in_leaf))

            opt.constraints += [rbi[leaf_in_prev_beam + leaf] - cp.multiply(c[1] + 1, z[beamlets_in_leaf]) >= 1]
            opt.constraints += [
                cp.multiply((beam['num_cols'] - c[1]), z[beamlets_in_leaf]) + lbi[leaf_in_prev_beam + leaf] <= beam[
                    'num_cols']]
            opt.constraints += [cp.sum([z[b_i] for b_i in beamlets_in_leaf]) == rbi[leaf_in_prev_beam + leaf] - lbi[
                leaf_in_prev_beam + leaf] - 1]
            opt.constraints += [rbi[leaf_in_prev_beam + leaf] <= beam['num_cols']]

            opt.constraints += [x[beamlets_in_leaf] <= U * z[beamlets_in_leaf]]
            opt.constraints += [mu[i] - U * (1 - z[beamlets_in_leaf]) <= x[beamlets_in_leaf]]
            opt.constraints += [x[beamlets_in_leaf] <= mu[i]]

        leaf_in_prev_beam = leaf_in_prev_beam + len(leaf_idx_beamlet_map)
        # bound constraints
        opt.constraints += [mu[i] <= U]
    opt.constraints += [lbi >= 0]
    opt.constraints += [rbi >= 0]
    # Optimize the problem
    sol = opt.solve(solver=cp.MOSEK, verbose=True)
    sol['MU'] = mu.value
    sol['left_leaf_pos'] = lbi.value
    sol['right_leaf_pos'] = rbi.value

    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol, sol_name='sol_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # my_plan = pp.load_plan(plan_name='my_plan_phantom_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # sol = pp.load_optimal_sol(sol_name='sol_phantom_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))

    """
    2) Visualize the benchmarked VMAT plan
    
    """
    # plot dvh for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    pp.Visualization.plot_dvh(my_plan, sol=sol, struct_names=struct_names, title=data.patient_id)

    # plot 2d axial slice for the given solution and display the structures contours on the slice
    pp.Visualization.plot_2d_slice(my_plan=my_plan, sol=sol, slice_num=60, struct_names=['PTV'])

    """
    3) Benchmarking Column generation and Column generation + SCP based VMAT against MIP based global optimal 
    
    """
    # Loading hyper-parameter values for optimization problem
    protocol_name = 'Lung_2Gy_30Fx_vmat'
    vmat_opt_params = data.load_config_opt_params(protocol_name=protocol_name)
    # remove aperture smoothness objectives for fair comparison with global optimal. Last two obj as smoothness objectives
    del vmat_opt_params['objective_functions'][-1]
    del vmat_opt_params['objective_functions'][-1]
    vmat_opt_params['opt_parameters']['smooth_delta'] = 0
    vmat_opt_params['opt_parameters']['mu_max'] = 1000
    vmat_opt_params['objective_functions'][1]['weight'] = 30000

    # Assign discrete beam/control_point_ids to arcs and create arcs dictionary.
    # Below is an example of creating 2 arcs. Users can create single or multiple arcs.
    arcs_dict = {'arcs': [{'arc_id': "01", "beam_ids": list(np.arange(0, 72, 11))}]}
    # Create an object of Arcs class
    arcs = pp.Arcs(arcs_dict=arcs_dict, inf_matrix=inf_matrix_db)

    # Create a plan using ct, structures, beams and influence matrix. Clinical criteria is optional
    my_plan_scp = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix_db,
                          clinical_criteria=clinical_criteria, arcs=arcs)

    # Create VMAT plan using column generation
    start_col_gen_time = time.time()
    if vmat_opt_params['opt_parameters']['initial_leaf_pos'].lower() == 'cg':
        vmat_opt_col_gen = pp.VmatOptimizationColGen(my_plan=my_plan_scp,
                                                     opt_params=vmat_opt_params)

        sol_col_gen = vmat_opt_col_gen.run_col_gen_algo(solver='MOSEK', verbose=False)
    end_col_gen_time = time.time() - start_col_gen_time
    print('Total column generation time: {}'.format(end_col_gen_time))

    # Create VMAT plan using column generation + SCP based approach
    # Create vmat optimization
    start_scp_time = time.time()
    vmat_opt = pp.VmatScpOptimization(my_plan=my_plan_scp,
                                      opt_params=vmat_opt_params)
    # Run Sequential convex algorithm for optimising the plan. The final result will be stored in sol and convergence will store the convergence history (i.e., results of each iteration)
    convergence = vmat_opt.run_sequential_cvx_algo(solver='MOSEK', verbose=False)
    end_scp_time = time.time() - start_scp_time
    print('Total SCP time: {}'.format(end_scp_time))
    sol_scp = convergence[vmat_opt.best_iteration]  # get best solution
    sol_scp['inf_matrix'] = inf_matrix_db  # reference to influence matrix object for further evaluation and visualization

    # compare different VMAT techniques compared to MIP based blobal optimal
    d = inf_matrix_db.A @ sol_col_gen['optimal_intensity']
    col_gen_objective_value = compute_objective(my_plan_scp, d, vmat_opt.obj_funcs, prescription_dose=2)
    print("Total Col Gen objective value:", col_gen_objective_value)

    # SCP
    d = inf_matrix_db.A @ sol_scp['optimal_intensity']
    scp_objective_value = compute_objective(my_plan_scp, d, vmat_opt.obj_funcs, prescription_dose=2)
    print("Total SCP objective value:", scp_objective_value)

    # MIP
    d = inf_matrix_db.A @ sol['optimal_intensity']
    global_objective_value = compute_objective(my_plan_scp, d, vmat_opt.obj_funcs, prescription_dose=2)
    print("Total Global Optimal objective value:", global_objective_value)

    import matplotlib.pyplot as plt
    sol_scp['inf_matrix'] = inf_matrix_db
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNG_R']
    fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
    sol_col_gen['inf_matrix'] = inf_matrix_db
    ax = pp.Visualization.plot_dvh(my_plan_scp, sol=sol, struct_names=struct_names, style='solid', ax=ax,
                                   norm_flag=True)
    ax = pp.Visualization.plot_dvh(my_plan_scp, sol=sol_col_gen, struct_names=struct_names, style='dashed', ax=ax,
                                   norm_flag=True)
    ax = pp.Visualization.plot_dvh(my_plan_scp, sol=sol_scp, struct_names=struct_names, style='dotted', ax=ax,
                                   norm_flag=True)
    ax.set_title('Solid: MIP Global Optimal  Dashed: Column generation  Dotted: Column generation + SCP')
    plt.show()

if __name__ == "__main__":
    vmat_optimization()
