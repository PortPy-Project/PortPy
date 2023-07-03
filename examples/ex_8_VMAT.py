"""

This example demonstrates the use of portpy.photon to create down sampled influence matrix and
    create a benchmark VMAT plan
    1- Down sample influence matrix and create a benchmark VMAT plan using few beams since it computationally expensive
    2- Visualize benchmark VMAT plan
    3- Evaluate the respective plan


"""
from portpy.photon as pp
import numpy as np
import cvxpy as cp
import os


def ex_8_VMAT():
    """
        1) Create down sampled influence matrix and generate benchmark VMAT plan

    """
    # Create plan object
    data_dir = r'../data'
    data = pp.DataExplorer(data_dir=data_dir)
    patient_id = 'Lung_Phantom_Patient_1'
    data.patient_id = patient_id

    # Load ct, structure and beams as an object
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beam_ids = [0, 9, 18]
    beams = pp.Beams(data, beam_ids=beam_ids)

    # create rinds based upon rind definition in optimization params
    opt_params = data.load_config_opt_params(protocol_name='Lung_2Gy_30Fx')
    structs.create_opt_structures(opt_params)

    # load influence matrix based upon beams and structure set
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # create a influence matrix down sampled beamlets of width and height 10mm
    beamlet_down_sample_factor = 4
    new_beamlet_width_mm = beams.get_finest_beamlet_width() * beamlet_down_sample_factor
    new_beamlet_height_mm = beams.get_finest_beamlet_height() * beamlet_down_sample_factor
    inf_matrix_db = inf_matrix.create_down_sample(beamlet_width_mm=new_beamlet_width_mm,
                                                  beamlet_height_mm=new_beamlet_height_mm)

    # load clinical criteria from the config files for which plan to be optimized
    protocol_name = 'Lung_2Gy_30Fx'
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name)

    my_plan = pp.Plan(ct, structs, beams, inf_matrix_db, clinical_criteria)

    # remove smoothness objective
    for i in range(len(opt_params['objective_functions'])):
        if opt_params['objective_functions'][i]['type'] == 'smoothness-quadratic':
            opt_params['objective_functions'][i]['weight'] = 0

    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()

    # data preprocessing for creating VMAT beams
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

    # create new variables for VMAT
    lbi = cp.Variable(total_rows, integer=True)
    rbi = cp.Variable(total_rows, integer=True)
    z = cp.Variable(my_plan.inf_matrix.A.shape[1], boolean=True)
    mu = cp.Variable(len(vmat_beams), pos=True)
    x = opt.vars['x']

    # Add geometric constraints (consecutive ones)
    leaf_in_prev_beam = 0
    for i, beam in enumerate(vmat_beams):
        beam_map = beam['beam_map']
        leaf_idx_beamlet_map = beam['leaf_idx_beamlet_map']

        for leaf in leaf_idx_beamlet_map:
            beamlets_in_leaf = leaf_idx_beamlet_map[leaf]
            ind = np.where(np.isin(beam_map, beamlets_in_leaf))
            # geometric constraints
            opt.constraints += [rbi[leaf_in_prev_beam + leaf] - cp.multiply(ind[1] + 1, z[beamlets_in_leaf]) >= 1]
            opt.constraints += [
                cp.multiply((beam['num_cols'] - ind[1]), z[beamlets_in_leaf]) + lbi[leaf_in_prev_beam + leaf] <= beam[
                    'num_cols']]
            opt.constraints += [cp.sum([z[b_i] for b_i in beamlets_in_leaf]) == rbi[leaf_in_prev_beam + leaf] - lbi[
                leaf_in_prev_beam + leaf] - 1]
            opt.constraints += [rbi[leaf_in_prev_beam + leaf] <= beam['num_cols']]
            # McCormick Constraints
            opt.constraints += [x[beamlets_in_leaf] <= 2 * z[beamlets_in_leaf]]
            opt.constraints += [mu[i] - 2 * (1 - z[beamlets_in_leaf]) <= x[beamlets_in_leaf]]
            opt.constraints += [x[beamlets_in_leaf] <= mu[i]]

        leaf_in_prev_beam = leaf_in_prev_beam + len(leaf_idx_beamlet_map)
        # bound constraints
        opt.constraints += [mu[i] <= 2]
    opt.constraints += [lbi >= 0]
    opt.constraints += [rbi >= 0]
    sol = opt.solve(solver=cp.MOSEK, verbose=True)
    sol['MU'] = mu.value
    sol['left_leaf_pos'] = lbi.value
    sol['right_leaf_pos'] = rbi.value

    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan_phantom_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    pp.save_optimal_sol(sol, sol_name='sol_phantom_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # my_plan = pp.load_plan(plan_name='my_plan_phantom_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))
    # sol = pp.load_optimal_sol(sol_name='sol_phantom_vmat.pkl', path=os.path.join(r'C:\temp', data.patient_id))

    """
    2) Visualize the benchmarked VMAT plan
    
    """
    # plot dvh for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    pp.Visualization.plot_dvh(my_plan, sol=sol, struct_names=struct_names, title=patient_id)

    # plot 2d axial slice for the given solution and display the structures contours on the slice
    pp.Visualization.plot_2d_slice(my_plan=my_plan, sol=sol, slice_num=60, struct_names=['PTV'])

    """
    3) Evaluate the plan
    
    """
    # visualize plan metrics based upon clinical criteria
    pp.Evaluation.display_clinical_criteria(my_plan, sol=sol)

    # view ct, dose_1d and segmentations in 3d slicer. This requires downloading and installing 3d slicer
    # First save the Nrrd images in data_dir directory
    pp.save_nrrd(my_plan, sol=sol, data_dir=os.path.join(r'C:\temp', data.patient_id))
    pp.Visualization.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 5.2.1\Slicer.exe',
                                    data_dir=os.path.join(r'C:\temp', data.patient_id))
    print('Done!')


if __name__ == "__main__":
    ex_8_VMAT()
