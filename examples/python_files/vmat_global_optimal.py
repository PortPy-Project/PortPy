"""

This example demonstrates the use of portpy.photon to create down sampled influence matrix and
create a benchmark VMAT plan

"""
import portpy.photon as pp
import numpy as np
import cvxpy as cp
import os
import time

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

# Solving this MIP takes many hours, so by default the example scores the PUBLISHED reference
# solution instead of re-solving. Set True to solve it yourself.
RUN_MIP = False
# Used when RUN_MIP is False. A bare file name is downloaded from the published artifacts
# (github.com/PortPy-Project/artifacts); a local .npz path is used as it is. The solution only
# applies to the beams and down-sampling set up below, so re-solve if you change them.
MIP_SOLUTION = 'vmat_mip_Lung_Patient_6.npz'


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
    if RUN_MIP:
        # Optimize the problem
        sol = opt.solve(solver=cp.MOSEK, verbose=True)
        sol['MU'] = mu.value
        sol['left_leaf_pos'] = lbi.value
        sol['right_leaf_pos'] = rbi.value

        # Comment/Uncomment these lines to save the pickle file for plans and optimal solution from the directory
        pp.save_plan(my_plan, plan_name='my_plan_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
        pp.save_optimal_sol(sol, sol_name='sol_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    else:
        # Score the published reference solution instead of re-solving. The constraints built
        # above are still what define the problem; this is simply its recorded answer.
        sol_path = (MIP_SOLUTION if os.path.exists(MIP_SOLUTION)
                    else str(pp.download_artifact(MIP_SOLUTION)))
        with np.load(sol_path) as npz:
            sol = {key: npz[key] for key in npz.files if not key.startswith('meta_')}
        n_beamlets = my_plan.inf_matrix.A.shape[1]
        if sol['optimal_intensity'].size != n_beamlets:
            raise ValueError(
                f"{os.path.basename(sol_path)} holds {sol['optimal_intensity'].size} beamlet "
                f'intensities but this plan has {n_beamlets}. A saved solution only applies to '
                f'the geometry it was solved for, so set RUN_MIP = True to re-solve, or restore '
                f'the beam selection and down-sampling the solution was made for.')
        # The published file carries only the arrays, not the influence matrix it was solved on
        # (that would make it a gigabyte). It is rebuilt above, and plot_dvh / plot_2d_slice
        # read it from here.
        sol['inf_matrix'] = my_plan.inf_matrix
        print(f'MIP loaded from {sol_path}')

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
    vmat_opt_params['opt_parameters']['mu_min'] = 0
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

    """
    4) Plan score -- does the objective difference matter clinically?

    """
    # The objective says which method optimized best. The scorecard says whether that
    # difference matters: it awards points per dose-volume criterion the way a planner does
    # when accepting or rejecting a plan, and Evaluation.get_plan_score sums them. Criteria
    # for structures this patient does not have are skipped, so the maximum attainable total
    # is patient specific. The scorecard shipped with PortPy is used here; users can supply
    # their own JSON with custom criteria and weights.
    import json

    alias_map = {
        "BILAT_FEM_HEADS": "FEMURS",
        "SPINAL_CORD": "CORD",
        "TOTAL LUNG - GTV": "LUNGS_NOT_GTV"
    }
    scorecard_path = os.path.join(os.path.dirname(os.path.dirname(pp.__file__)),
                                  'config_files', 'scorecard',
                                  'SC_Lung(60Gy)_2022MAAS_ExampleV2.json')
    with open(scorecard_path, "r") as f:
        scorecard_json = json.load(f)

    # The plans are OPTIMIZED on the voxel-down-sampled matrix but SCORED on the full voxel
    # grid, because the scorecard is a clinical read-out and coarse voxels misrepresent it:
    # at 6x down-sampling the PTV has 1,123 voxels instead of 3,954, which under-resolves
    # the target's cold tail and costs ~4 points on the minimum-dose criterion alone (the
    # organ-at-risk criteria are unaffected). That needs a matrix down-sampled in beamlets
    # only, plus a plan built on it -- passing a full-grid dose together with the
    # voxel-down-sampled plan would index the structures with the wrong voxel set.
    inf_matrix_full = inf_matrix.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                    beamlet_height_mm=new_beamlet_height_mm)
    plan_full = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix_full,
                        clinical_criteria=clinical_criteria)

    print('\n\n%-30s %12s' % ('method', 'plan score'))
    print('-' * 44)
    score_df = None
    for name, intensity in (('MIP (global optimal)', sol['optimal_intensity']),
                            ('Column generation', sol_col_gen['optimal_intensity']),
                            ('Column generation + SCP', sol_scp['optimal_intensity'])):
        dose_1d = inf_matrix_full.A @ intensity * my_plan.get_num_of_fractions()
        # Normalize each plan so the target gets the prescription at D90, as plans are
        # compared in the clinic: without it a plan that simply delivers less dose
        # everywhere would score better on every organ-at-risk criterion.
        dummy = {'optimal_intensity': intensity, 'inf_matrix': inf_matrix_full}
        norm_factor = pp.Evaluation.get_dose(dummy, dose_1d=dose_1d, struct='PTV',
                                             volume_per=90) / my_plan.get_prescription()
        score, score_df = pp.Evaluation.get_plan_score(
            plan=plan_full,
            dose_1d=dose_1d / norm_factor,
            scorecard_json=scorecard_json,
            prescription_gy=my_plan.get_prescription(),
            alias_map=alias_map,
            return_score_df=True
        )
        print('%-30s %12.2f' % (name, score))

    # criterion by criterion for the last method scored, so a total can always be traced
    # back to the criteria that produced it
    if score_df is not None:
        print()
        print(score_df[['Structure', 'MetricType', 'MetricValue', 'Score',
                        'Status']].to_string(index=False))


if __name__ == "__main__":
    vmat_optimization()
