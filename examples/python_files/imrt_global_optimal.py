"""
Four ways to build a DELIVERABLE step-and-shoot IMRT plan, compared head to head.

IMRT step-and-shoot delivers K apertures per beam, each with its own MU, and a
beamlet's fluence is the SUM of the apertures that leave it open. Getting from a
dose objective to apertures can be done in very different ways, and this example
runs all four on ONE problem so the trade-offs are visible:

1. FMO + leaf sequencing + MU re-optimization -- the conventional two-stage
   route. Solve the continuous fluence-map problem (with the smoothness term, so
   the fluence is one an MLC can follow), convert it to leaf positions with
   Siochi stratification, then keep the aperture shapes and re-solve the segment
   MUs. The MU step repairs the stratification error; without it the route is not
   a fair comparison. Deliverability is enforced AFTER the fact.
2. ColGen -- column generation: add one aperture at a time, each chosen against
   the dose still missing. Deliverable by construction.
3. ColGen + SCP -- then refine the leaf positions with sequential convex
   programming (``portpy.photon.imrt_scp``). Deliverable and fast.
4. MIP -- direct aperture optimization, with the MLC constraints inside the
   optimization. The global-optimal reference, but slow: it needs days and still
   does not prove optimality here.

MIP formulation, per beam i and aperture k:

    z[i,k,b] in {0,1}   beamlet b is open in aperture k
    l[i,k,r], u[i,k,r]  left / right leaf position in row r   (integer)
    mu[i,k] >= 0        monitor units of that aperture

  consecutive-ones (an open row is one contiguous run between the leaves)
  and, because x = mu * z is bilinear, the standard big-M linearization

    y[i,k,b] <= U * z[i,k,b]
    y[i,k,b] <= mu[i,k]
    y[i,k,b] >= mu[i,k] - U * (1 - z[i,k,b])
    x[b]      = sum_k y[i,k,b]

The continuous FMO fluence is NOT reported as a method. No MLC can deliver it, so
it is not a plan -- it is only the input to leaf sequencing in method 1. Every row
of the table is a plan a machine could treat with, and the MIP is the reference
they are measured against.

FAIRNESS -- all four solve the same problem and are scored the same way:
  * one shared setup (same patient, beams, down-sampling, opt structures);
  * the same aperture budget, ``APERTURES_PER_BEAM`` per beam. The leaf sequencing
    level is swept, because its segment count is an output, and the row matching
    that budget is the one to read;
  * all four MINIMIZE the same objective. The dose terms are linear everywhere
    (the ``linear-*`` objective types), so no method is being scored on a function
    it did not optimize;
  * every plan is then scored with ONE function, :func:`linear_objective_value`;
  * the aperture-regularity and aperture-similarity terms are dropped, because the
    MIP does not model them -- leaving them on would penalise the SCP for
    something its competitor is not asked to do.

The MIP is only tractable on a reduced problem, so the influence matrix is
down-sampled in both voxels and beamlets and only a few beam angles are used.
Every knob that controls the size is in USER SETTINGS below.
"""
import os
import sys
import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

import portpy.photon as pp
from portpy.photon.imrt_scp import (create_imrt_plan, ImrtOptimizationColGen,
                                    ImrtScpOptimization)

# =========================================================
# USER SETTINGS -- these control how big the MIP gets
# =========================================================
DATA_DIR = r'../../data'
PATIENT_ID = 'Lung_Patient_6'
PROTOCOL_NAME = 'Lung_2Gy_30Fx'

# A full IMRT plan uses 7-9 beams; the MIP grows quickly, so start small.
NUM_BEAMS = 5
# Apertures per beam. Clinical step-and-shoot uses 5-10; 3 keeps this runnable.
APERTURES_PER_BEAM = 10

# Down-sampling, same idea as the VMAT example. Binary count scales with
# (beamlets / factor^2) * NUM_BEAMS * APERTURES_PER_BEAM, so raising the factor
# is the fastest way to make this finish.
VOXEL_DOWN_SAMPLE = [6, 6, 1]
BEAMLET_DOWN_SAMPLE = 6

U = 2.0                 # upper bound on the MU of a single aperture (MIP)
SCP_MU_MAX = 1000
# Quadratic dose objective -> its linear counterpart, so every method optimizes the same
# objective. Types not listed here are dropped, because the MIP does not model them.
LINEARISE = {'quadratic-overdose': 'linear-overdose',
             'quadratic-underdose': 'linear-underdose',
             'quadratic': 'linear'}
MIP_TIME_LIMIT_S = 180000000  # wall-clock cap so the example always terminates
# Stratification levels for Siochi leaf sequencing, swept rather than fixed.
#
# WHY A SWEEP: the MIP and the SCP are both capped at APERTURES_PER_BEAM
# apertures per beam, but Siochi's segment count is NOT a free parameter -- you
# set the number of stratification levels and the segments fall out of the
# fluence. A single arbitrary level count (40, say) hands the two-stage route far
# more segments than its competitors and the comparison stops meaning anything.
# So sweep it, report the SEGMENT COUNT for every method, and read the row whose
# segment count is closest to NUM_BEAMS * APERTURES_PER_BEAM as the matched
# comparison. The rest of the sweep is informative in its own right: it shows how
# many segments the two-stage route needs to reach a given plan quality.
LEAF_SEQ_LEVELS = (2, 3, 5, 10, 20, 40)
# Solving the MIP takes days at a clinical aperture count, so by default this example scores
# the PUBLISHED reference solution instead of re-solving. Set True to solve it yourself.
RUN_MIP = False
# Used when RUN_MIP is False. A bare file name is downloaded from the published artifacts
# (github.com/PortPy-Project/artifacts); a local .npz or .pkl path is used as it is; '' leaves
# the MIP out of the comparison entirely.
#
# The published solution was solved for the settings above, so if you change PATIENT_ID,
# NUM_BEAMS, APERTURES_PER_BEAM or the down-sampling you must re-solve with RUN_MIP = True --
# loading it anyway raises rather than silently scoring the wrong plan.
MIP_SOLUTION = 'imrt_mip_Lung_Patient_6.npz'
MIP_SOLVER = cp.GUROBI
MIP_PARAMS = dict(TimeLimit=MIP_TIME_LIMIT_S,
                  Cuts=2,          # aggressive cut generation
                  Symmetry=2,      # K! aperture permutations per beam
                  MIPFocus=1,      # favour finding better incumbents over proving bounds
                  Heuristics=0.2)
# Everything printed -- including every solver iteration -- is teed to this file.
LOG_FILE = 'imrt_global_optimal_log.txt'
# Passed to every solve(). True prints the full solver iteration log.
VERBOSE_SOLVER = True
SAVE_DIR = r'C:\temp'
STRUCTS_TO_PLOT = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD', 'LUNGS_NOT_GTV']


class Tee:
    """Write to the console and to the log file at once.

    cvxpy routes the MOSEK iteration log through Python's stdout, so replacing
    stdout captures the solver output as well as our own prints.
    """

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()


def build_beam_maps(inf_matrix, beam_ids):
    """Per beam, the BEV grid and the beamlet indices in each leaf row."""
    beam_maps = inf_matrix.get_bev_2d_grid(beam_id=beam_ids)
    beams_info = []
    for beam_map in beam_maps:
        # drop rows that are entirely outside the BEV
        beam_map = beam_map[~np.all(beam_map == -1, axis=1), :]
        num_rows, num_cols = beam_map.shape
        row_beamlets = {r: row[row >= 0].tolist() for r, row in enumerate(beam_map)}
        beams_info.append({'beam_map': beam_map, 'num_rows': num_rows,
                           'num_cols': num_cols, 'row_beamlets': row_beamlets})
    return beams_info


def add_aperture_constraints(opt, beams_info, x, n_beamlets):
    """MLC geometry + big-M linearization. Returns (z, mu, y) variable lists."""
    n_ap = len(beams_info) * APERTURES_PER_BEAM
    total_rows = sum(b['num_rows'] for b in beams_info) * APERTURES_PER_BEAM

    z = cp.Variable((n_ap, n_beamlets), boolean=True)
    y = cp.Variable((n_ap, n_beamlets), nonneg=True)
    mu = cp.Variable(n_ap, nonneg=True)
    lbi = cp.Variable(total_rows, integer=True)
    rbi = cp.Variable(total_rows, integer=True)

    row_offset = 0
    for i, beam in enumerate(beams_info):
        beam_map = beam['beam_map']
        beam_beamlets = sorted(
            {b for bl in beam['row_beamlets'].values() for b in bl})

        for k in range(APERTURES_PER_BEAM):
            a = i * APERTURES_PER_BEAM + k          # flat aperture index

            for leaf, beamlets in beam['row_beamlets'].items():
                if not beamlets:
                    continue
                r = row_offset + leaf
                c = np.where(np.isin(beam_map, beamlets))   # column positions

                # consecutive ones: open beamlets form one run between the leaves
                opt.constraints += [
                    rbi[r] - cp.multiply(c[1] + 1, z[a, beamlets]) >= 1,
                    cp.multiply(beam['num_cols'] - c[1], z[a, beamlets])
                    + lbi[r] <= beam['num_cols'],
                    cp.sum(z[a, beamlets]) == rbi[r] - lbi[r] - 1,
                    rbi[r] <= beam['num_cols'],
                ]

                # y = mu * z, linearized (bilinear term)
                opt.constraints += [
                    y[a, beamlets] <= U * z[a, beamlets],
                    y[a, beamlets] <= mu[a],
                    y[a, beamlets] >= mu[a] - U * (1 - z[a, beamlets]),
                ]

            # beamlets of this beam never opened by this aperture carry no MU
            closed = [b for b in range(n_beamlets) if b not in beam_beamlets]
            if closed:
                opt.constraints += [z[a, closed] == 0, y[a, closed] == 0]

            opt.constraints += [mu[a] <= U]
            row_offset += beam['num_rows']

        # IMRT fluence = SUM over this beam's apertures (the VMAT case is K=1)
        aps = [i * APERTURES_PER_BEAM + k for k in range(APERTURES_PER_BEAM)]
        opt.constraints += [x[beam_beamlets]
                            == cp.sum(y[aps, :], axis=0)[beam_beamlets]]

        # Symmetry breaking. The apertures of one beam are interchangeable, so
        # every solution has K! relabellings with the same objective and the
        # branch-and-bound tree explores all of them. Ordering the MU keeps one
        # representative per class without removing any achievable plan.
        opt.constraints += [mu[aps[k]] >= mu[aps[k + 1]]
                            for k in range(APERTURES_PER_BEAM - 1)]

    opt.constraints += [lbi >= 0, rbi >= 0]
    return z, mu, y


def linear_objectives(opt, opt_params, my_plan, inf_matrix, clinical_criteria):
    """Replace the quadratic objectives with linear ones, as the VMAT MIP does
    (a quadratic objective would make this a MIQP and blow up the solve)."""
    obj_funcs = opt_params.get('objective_functions', [])
    A = inf_matrix.A
    st = inf_matrix
    x = opt.vars['x']
    opt.obj = []

    for obj in obj_funcs:
        struct = obj.get('structure_name')
        if struct not in my_plan.structures.get_structures():
            continue
        idx = st.get_opt_voxels_idx(struct)
        if len(idx) == 0:
            continue

        if obj['type'] == 'quadratic-overdose':
            dose_gy = opt.get_num(obj['dose_gy']) / clinical_criteria.get_num_of_fractions()
            dO = cp.Variable(len(idx), pos=True)
            opt.obj += [(1 / len(idx)) * (obj['weight'] * cp.sum(dO))]
            opt.constraints += [A[idx, :] @ x <= dose_gy + dO]
        elif obj['type'] == 'quadratic-underdose':
            dose_gy = opt.get_num(obj['dose_gy']) / clinical_criteria.get_num_of_fractions()
            dU = cp.Variable(len(idx), pos=True)
            opt.obj += [(1 / len(idx)) * (obj['weight'] * cp.sum(dU))]
            opt.constraints += [A[idx, :] @ x >= dose_gy - dU]
        elif obj['type'] == 'quadratic':
            opt.obj += [(1 / len(idx)) * (obj['weight'] * cp.sum(A[idx, :] @ x))]


def linear_objective_value(dose_1d, opt_params, my_plan, inf_matrix,
                           clinical_criteria, get_num):
    """Score ANY plan with the objective :func:`linear_objectives` minimizes.

    This is the single yardstick every method is measured with. It evaluates the
    same terms in closed form: the over/underdose slacks are free positive
    variables that are minimized, so at the optimum ``dO = max(0, d - dose_gy)``
    and ``dU = max(0, dose_gy - d)``.

    :param dose_1d: dose PER FRACTION on the optimization voxels
t    :param get_num: ``Optimization.get_num``, which resolves "prescription_gy"
        expressions; passed in so this does not re-implement that lookup
    :return: objective value, directly comparable across methods
    """
    total = 0.0
    num_fractions = clinical_criteria.get_num_of_fractions()
    for obj in opt_params.get('objective_functions', []):
        struct = obj.get('structure_name')
        if struct not in my_plan.structures.get_structures():
            continue
        idx = inf_matrix.get_opt_voxels_idx(struct)
        if len(idx) == 0:
            continue
        d = dose_1d[idx]
        if obj['type'] == 'quadratic-overdose':
            gy = get_num(obj["dose_gy"]) / num_fractions
            total += obj['weight'] / len(idx) * float(np.sum(np.maximum(0.0, d - gy)))
        elif obj['type'] == 'quadratic-underdose':
            gy = get_num(obj["dose_gy"]) / num_fractions
            total += obj['weight'] / len(idx) * float(np.sum(np.maximum(0.0, gy - d)))
        elif obj['type'] == 'quadratic':
            total += obj['weight'] / len(idx) * float(np.sum(d))
    return total


def imrt_global_optimal():
    data = pp.DataExplorer(data_dir=DATA_DIR)
    data.patient_id = PATIENT_ID

    ct = pp.CT(data)
    structs = pp.Structures(data)

    # A handful of equidistant angles. IMRT is already a few-beam modality, so
    # unlike VMAT this is a realistic plan rather than a coarse approximation.
    all_ids = data.load_metadata()['beams']['ID']
    step = max(1, len(all_ids) // NUM_BEAMS)
    beam_ids = list(all_ids[::step])[:NUM_BEAMS]
    print(f'Using {len(beam_ids)} beams: {beam_ids}')
    beams = pp.Beams(data, beam_ids=beam_ids)

    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=PROTOCOL_NAME)
    opt_params = data.load_config_opt_params(protocol_name=PROTOCOL_NAME)
    structs.create_opt_structures(opt_params=opt_params,
                                  clinical_criteria=clinical_criteria)

    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    opt_vox_xyz_res_mm = [r * f for r, f in
                          zip(ct.get_ct_res_xyz_mm(), VOXEL_DOWN_SAMPLE)]
    inf_matrix_db = inf_matrix.create_down_sample(
        beamlet_width_mm=beams.get_finest_beamlet_width() * BEAMLET_DOWN_SAMPLE,
        beamlet_height_mm=beams.get_finest_beamlet_height() * BEAMLET_DOWN_SAMPLE,
        opt_vox_xyz_res_mm=opt_vox_xyz_res_mm)
    print(f'Down-sampled A: {inf_matrix_db.A.shape} '
          f'(voxels x beamlets)')

    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams,
                      inf_matrix=inf_matrix_db,
                      clinical_criteria=clinical_criteria)

    # Save as we go, not at the end: the MIP can run for hours at a realistic
    # aperture count, and if it is stopped everything before it is already on disk.
    # Reload with pp.load_plan / np.load to redo the tables and DVHs without
    # re-solving anything.
    save_dir = os.path.join(SAVE_DIR, data.patient_id)
    os.makedirs(save_dir, exist_ok=True)
    pp.save_plan(my_plan, plan_name='my_plan_imrt.pkl', path=save_dir)

    # Smoothness is dropped from the objective every method is SCORED on, because the MIP
    # does not model it. Keep its weight: the fluence sent to leaf sequencing IS smoothed.
    smooth_weight = next((o['weight'] for o in opt_params['objective_functions']
                          if o['type'] == 'smoothness-quadratic'), 0.0)
    for obj in opt_params['objective_functions']:
        if obj['type'] == 'smoothness-quadratic':
            obj['weight'] = 0

    # ---------------------------------------------------------
    # 1) Continuous fluence-map optimization (FMO)
    # ---------------------------------------------------------
    # The smoothness term is added back here, because this fluence is the one leaf
    # sequencing has to follow: a spiky map needs many stratification levels, which costs
    # both accuracy and segments. It is NOT part of the objective methods are scored on.
    opt_fmo = pp.Optimization(my_plan, opt_params=opt_params)
    opt_fmo.create_cvxpy_problem()
    linear_objectives(opt_fmo, opt_params, my_plan, inf_matrix_db, clinical_criteria)
    Qx, Qy, sm_rows, sm_cols = pp.Optimization.get_smoothness_matrix(
        inf_matrix_db.beamlets_dict)
    opt_fmo.obj += [smooth_weight
                    * (0.6 * (1 / sm_cols) * cp.sum_squares(Qx @ opt_fmo.vars['x'])
                       + 0.4 * (1 / sm_rows) * cp.sum_squares(Qy @ opt_fmo.vars['x']))]
    t0 = time.time()
    sol_fmo = opt_fmo.solve(solver=cp.MOSEK, verbose=VERBOSE_SOLVER)
    t_fmo = time.time() - t0
    print(f'\nFMO (continuous, smoothness weight {smooth_weight}): {t_fmo:.1f} s')

    num_fractions = clinical_criteria.get_num_of_fractions()
    A = inf_matrix_db.A
    ptv = inf_matrix_db.get_opt_voxels_idx('PTV')
    aperture_budget = len(beam_ids) * APERTURES_PER_BEAM
    # every method ends up as one (name, dose, seconds, segments) row
    results = []
    pp.save_optimal_sol(sol_fmo, sol_name='sol_fmo.pkl', path=save_dir)

    # ---------------------------------------------------------
    # 2) FMO + leaf sequencing + segment MU re-optimization
    # ---------------------------------------------------------
    # leaf sequencing gives back the fluence that is actually DELIVERABLE after
    # stratification, so A @ w_approx is the dose the patient would receive --
    # not the continuous fluence the optimizer asked for.
    print('\nFMO(smooth) + leaf sequencing (segment count is an OUTPUT, hence the sweep):')
    best_levels, best_dose, best_seg, best_time, best_seq = None, None, None, None, None
    for levels in LEAF_SEQ_LEVELS:
        t0 = time.time()
        seq = pp.leaf_sequencing_siochi(my_plan, sol_fmo, num_of_levels=levels)
        t_seq = time.time() - t0
        n_seg = sum(len(v['MU']) for k, v in seq.items() if isinstance(v, dict))
        dose = np.asarray(A @ seq['optimal_intensity'])
        obj = linear_objective_value(dose, opt_params, my_plan, inf_matrix_db,
                                     clinical_criteria, opt_fmo.get_num)
        print(f'  levels {levels:3d} -> {n_seg:4d} segments, '
              f'objective {obj:9.3f}, {t_seq:.1f} s')
        # keep the one whose segment count is closest to the MIP/SCP budget
        if best_seg is None or abs(n_seg - aperture_budget) < abs(best_seg - aperture_budget):
            best_levels, best_dose, best_seg = levels, dose, n_seg
            best_time, best_seq = t_fmo + t_seq, seq

    print(f'  -> matched to the {aperture_budget}-segment budget: '
          f'levels={best_levels} ({best_seg} segments)')
    np.save(os.path.join(save_dir, 'dose_fmo_leafseq.npy'), best_dose)
    # the sequencer output carries the leaf positions and MU of every segment
    pp.save_optimal_sol(best_seq, sol_name='sol_fmo_leafseq.pkl', path=save_dir)

    # Keep the aperture shapes from the sequencer and re-solve only the segment MUs, which
    # repairs the stratification error. Built as a pp.Optimization so it inherits the same
    # objective AND the same clinical constraints; `x == Z.T @ mu` keeps the fluence a sum of
    # those apertures, which is what makes the result deliverable.
    t0 = time.time()
    cols = []
    for beam_id in my_plan.beams.get_all_beam_ids():
        gantry_angle = my_plan.beams.get_gantry_angle(beam_id=beam_id)
        if gantry_angle not in best_seq:
            continue
        grid = inf_matrix_db.get_bev_2d_grid(beam_id=beam_id, finest_grid=False)
        for shape in best_seq[gantry_angle]['leaf_shapes']:
            idx = grid[np.asarray(shape) > 0]
            col = np.zeros(A.shape[1])
            col[idx[idx >= 0]] = 1.0
            cols.append(col)
    Z = np.asarray(cols)                      # (n_segments, n_beamlets), 1 = open
    opt_re = pp.Optimization(my_plan, opt_params=opt_params)
    opt_re.create_cvxpy_problem()
    linear_objectives(opt_re, opt_params, my_plan, inf_matrix_db, clinical_criteria)
    mu_re = cp.Variable(Z.shape[0], nonneg=True)
    opt_re.constraints += [opt_re.vars['x'] == Z.T @ mu_re]
    opt_re.solve(solver=cp.MOSEK, verbose=VERBOSE_SOLVER)
    w_re = np.maximum(np.asarray(mu_re.value).ravel(), 0.0)
    dose_re = np.asarray(A @ (Z.T @ w_re))
    n_seg_re = int(np.sum(w_re > 1e-9))
    print(f'MU re-optimization: {best_seg} segments -> {n_seg_re} with MU, '
          f'total MU {w_re.sum():.2f}, {time.time() - t0:.1f} s')
    results.append(('FMO + leaf seq', dose_re,
                    best_time + (time.time() - t0), n_seg_re))
    np.save(os.path.join(save_dir, 'dose_fmo_leafseq_muopt.npy'), dose_re)

    # ---------------------------------------------------------
    # 3) ColGen + SCP -- deliverable by construction, and fast
    # ---------------------------------------------------------
    vmat_params = data.load_config_opt_params(protocol_name=f'{PROTOCOL_NAME}_vmat')
    vmat_params['opt_parameters']['initial_leaf_pos'] = 'CG'
    vmat_params['opt_parameters']['mu_max'] = SCP_MU_MAX
    # Put ColGen and the SCP on the same objective as the MIP and the FMO. The VMAT
    # protocol states its dose objectives as quadratic, so swap in the linear types.
    # Without this each method optimizes a different function and the table means nothing.
    vmat_params['objective_functions'] = [
        dict(obj, type=LINEARISE[obj['type']])
        for obj in vmat_params['objective_functions'] if obj['type'] in LINEARISE]

    imrt_plan = create_imrt_plan(my_plan, num_apertures=APERTURES_PER_BEAM)
    t0 = time.time()
    cg_opt = ImrtOptimizationColGen(my_plan=imrt_plan, opt_params=vmat_params)
    sol_cg = cg_opt.run_col_gen_algo(solver='MOSEK', verbose=VERBOSE_SOLVER,
                                     accept_unknown=True)
    t_cg = time.time() - t0
    dose_cg = np.asarray(imrt_plan.inf_matrix.A @ sol_cg['optimal_intensity'])
    obj_cg = linear_objective_value(dose_cg, opt_params, my_plan, inf_matrix_db,
                                    clinical_criteria, opt_fmo.get_num)
    mu_cg = np.asarray(sol_cg['optimal_intensity'])
    n_ap_cg = int(np.sum([np.any(mu_cg[b['start_beamlet_idx']:
                                       b['end_beamlet_idx'] + 1] > 1e-9)
                          for b in imrt_plan.inf_matrix.beamlets_dict]))
    print(f'\nColGen seed: {t_cg:.1f} s, objective {obj_cg:.3f}, {n_ap_cg} segments with MU')
    results.append(('ColGen', dose_cg, t_cg, n_ap_cg))

    scp_opt = ImrtScpOptimization(my_plan=imrt_plan, opt_params=vmat_params)
    convergence = scp_opt.run_sequential_cvx_algo(solver='MOSEK', verbose=VERBOSE_SOLVER)
    t_scp = time.time() - t0
    sol_scp = convergence[scp_opt.best_iteration]
    dose_scp = np.asarray(sol_scp.get('best_act_dose_v', sol_scp['act_dose_v']))
    mu_scp = np.asarray(sol_scp['optimal_intensity'])
    n_ap_scp = int(np.sum([np.any(mu_scp[b['start_beamlet_idx']:
                                         b['end_beamlet_idx'] + 1] > 1e-9)
                           for b in imrt_plan.inf_matrix.beamlets_dict]))
    print(f'ColGen + SCP: {t_scp:.1f} s, {len(convergence)} SCP iterations, '
          f'{n_ap_scp} segments with MU')
    results.append(('ColGen + SCP', dose_scp, t_scp, n_ap_scp))
    np.save(os.path.join(save_dir, 'dose_colgen.npy'), dose_cg)
    np.save(os.path.join(save_dir, 'dose_colgen_scp.npy'), dose_scp)
    pp.save_optimal_sol(sol_cg, sol_name='sol_colgen.pkl', path=save_dir)
    # sol_scp['arcs'] holds the leaf positions and MU of every aperture
    pp.save_optimal_sol(sol_scp, sol_name='sol_colgen_scp.pkl', path=save_dir)

    # ---------------------------------------------------------
    # 4) Deliverability-constrained MIP -- the global-optimal reference
    # ---------------------------------------------------------
    sol_mip = None
    if RUN_MIP:
        opt_mip = pp.Optimization(my_plan, opt_params=opt_params)
        opt_mip.create_cvxpy_problem()
        linear_objectives(opt_mip, opt_params, my_plan, inf_matrix_db, clinical_criteria)

        beams_info = build_beam_maps(inf_matrix_db, my_plan.beams.get_all_beam_ids())
        n_beamlets = inf_matrix_db.A.shape[1]
        x = opt_mip.vars['x']
        z, mu, y = add_aperture_constraints(opt_mip, beams_info, x, n_beamlets)
        n_bin = len(beams_info) * APERTURES_PER_BEAM * n_beamlets
        print(f'\nMIP: {len(beams_info)} beams x {APERTURES_PER_BEAM} apertures, '
              f'{n_bin} binaries')

        t0 = time.time()
        sol_mip = opt_mip.solve(solver=MIP_SOLVER, verbose=VERBOSE_SOLVER, **MIP_PARAMS)
        t_mip = time.time() - t0
        sol_mip['MU'] = mu.value
        sol_mip['aperture_open'] = z.value
        print(f'MIP (deliverable): {t_mip:.1f} s')

        dose_mip = np.asarray(A @ sol_mip['optimal_intensity'])
        results.append(('MIP', dose_mip, t_mip, int(np.sum(mu.value > 1e-9))))
        np.save(os.path.join(save_dir, 'dose_mip.npy'), dose_mip)
        # sol_mip carries MU and the open/closed state of every beamlet per aperture
        pp.save_optimal_sol(sol_mip, sol_name='sol_mip.pkl', path=save_dir)
    elif MIP_SOLUTION:
        # Score a MIP that was solved earlier instead of re-solving it. It is the best plan
        # FOUND in the time the solver was given, not a proven optimum, so the table reports
        # its runtime as 'loaded'.
        if os.path.exists(MIP_SOLUTION):
            sol_path = MIP_SOLUTION
        else:
            sol_path = str(pp.download_artifact(MIP_SOLUTION))

        if sol_path.endswith('.npz'):
            with np.load(sol_path) as npz:
                sol_mip = {k: npz[k] for k in npz.files if not k.startswith('meta_')}
        else:
            sol_mip = pp.load_optimal_sol(sol_name=os.path.basename(sol_path),
                                          path=os.path.dirname(sol_path))

        intensity = np.asarray(sol_mip['optimal_intensity']).ravel()
        if intensity.size != A.shape[1]:
            raise ValueError(
                f'{os.path.basename(sol_path)} holds {intensity.size} beamlet intensities but '
                f'this plan has {A.shape[1]}. A saved solution only applies to the geometry it '
                f'was solved for, so re-solve with RUN_MIP = True, or restore the settings the '
                f'solution was made for.')
        dose_mip = np.asarray(A @ intensity)
        results.append(('MIP', dose_mip, float('nan'),
                        int(np.sum(np.asarray(sol_mip['MU']) > 1e-9))))
        print(f'\nMIP loaded from {sol_path}')

    # ---------------------------------------------------------
    # 5) Compare -- one objective, one table, one DVH
    # ---------------------------------------------------------
    objectives = [linear_objective_value(dose, opt_params, my_plan, inf_matrix_db,
                                         clinical_criteria, opt_fmo.get_num)
                  for _, dose, _, _ in results]
    # The MIP is the reference: it is the only method that states the machine exactly, so
    # the gap to it is what the fast methods are still leaving on the table.
    ref_i = next((i for i, (n, _, _, _) in enumerate(results) if n.startswith('MIP')), None)
    ref_obj = objectives[ref_i] if ref_i is not None else min(objectives)

    print('\n\n%-30s %10s %9s %6s %9s %8s %8s %8s'
          % ('method', 'objective', 'gap %', 'segs', 'time s', 'D95', 'mean', 'Dmax'))
    print('-' * 94)
    for (name, dose, seconds, segs), obj in zip(results, objectives):
        total = dose * num_fractions
        print('%-30s %10.3f %9.2f %6d %9s %8.2f %8.2f %8.2f'
              % (name, obj, 100 * (obj - ref_obj) / abs(ref_obj), segs,
                 '-' if seconds != seconds else '%.1f' % seconds,
                 np.percentile(total[ptv], 5), total[ptv].mean(), total.max()))
    print('\ngap %% is measured against the MIP, the best deliverable plan we can produce.')
    print('All objectives come from linear_objective_value(), so they are comparable.')

    # Plan score -- does the objective difference matter clinically?
    # The objective says which method optimized best. The scorecard says whether that
    # difference matters: it awards points per dose-volume criterion the way a planner does
    # when accepting or rejecting a plan. Each criterion's points are a piecewise-linear
    # interpolation of its tolerance bands at the achieved value, so a plan loses points
    # gradually rather than passing or failing a threshold. Criteria for structures this
    # patient does not have are skipped, so the maximum attainable total is patient specific.
    # Scored on the same down-sampled optimization voxels the methods were optimized on, so
    # the scores line up with the objective column above.
    import json

    # The scorecard ships with PortPy. This one is 2Gy x 30Fx = 60Gy, the same prescription
    # used here. For a different prescription, rescale it first with
    # portpy.photon.utils.adapt_scorecard_prescription.
    scorecard_path = os.path.join(os.path.dirname(os.path.dirname(pp.__file__)),
                                  'config_files', 'scorecard',
                                  'SC_Lung(60Gy)_2022MAAS_ExampleV2.json')
    with open(scorecard_path, 'r') as f:
        scorecard_json = json.load(f)

    # map scorecard structure names onto the names used in the PortPy plan
    alias_map = {'SPINAL_CORD': 'CORD', 'TOTAL LUNG - GTV': 'LUNGS_NOT_GTV'}

    print('\n\n%-30s %12s' % ('method', 'plan score'))
    print('-' * 44)
    score_df = None
    for name, dose, _, _ in results:
        score, score_df = pp.Evaluation.get_plan_score(
            plan=my_plan, dose_1d=dose * num_fractions, scorecard_json=scorecard_json,
            prescription_gy=my_plan.get_prescription(), alias_map=alias_map,
            return_score_df=True)
        print('%-30s %12.2f' % (name, score))

    # criterion by criterion for the last method scored, so a total can always be traced
    # back to the criteria that produced it
    if score_df is not None:
        print()
        print(score_df[['Structure', 'MetricType', 'MetricValue', 'Score',
                        'Status']].to_string(index=False))

    # one bar per method, as in the VMAT example. The dashed line is the MIP, and each bar
    # carries its gap to it, so the whole comparison reads off one picture.
    fig, ax_bar = plt.subplots(figsize=(6, 4.5))
    labels = [n.replace(' + ', ' +\n') for n, _, _, _ in results]
    bars = ax_bar.bar(labels, objectives,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(results)])
    for i, (bar, (_, _, seconds, segs)) in enumerate(zip(bars, results)):
        if seconds == seconds:                      # no time for a loaded solution
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        '%d s' % round(seconds), ha='center', va='bottom', fontsize=8)
        if i != ref_i:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.995,
                        '%+.1f%%' % (100 * (objectives[i] - ref_obj) / ref_obj),
                        ha='center', va='top', fontsize=8, color='white', fontweight='bold')
    ax_bar.axhline(ref_obj, ls='--', color='black', lw=0.8)
    ax_bar.set_ylabel('objective value')
    ax_bar.set_ylim(min(objectives) * 0.97, max(objectives) * 1.08)
    ax_bar.set_title(f'{data.patient_id}: {NUM_BEAMS} beams, '
                     f'{aperture_budget}-segment budget')
    plt.tight_layout()
    bar_path = os.path.join(save_dir, 'objective_imrt_comparison.png')
    plt.savefig(bar_path, dpi=300)
    print(f'objective comparison saved: {bar_path}')

    # DVH of the DELIVERABLE plans only -- the FMO fluence is not a plan, so it is left out.
    struct_names = [s for s in STRUCTS_TO_PLOT
                    if s in my_plan.structures.get_structures()]
    ax = None
    for (name, dose, _, _), style in zip(results, ('solid', 'dashed', 'dotted', 'dashdot')):
        ax = pp.Visualization.plot_dvh(my_plan, dose_1d=dose * num_fractions,
                                       struct_names=struct_names, style=style, ax=ax,
                                       show=False, norm_flag=True)
    ax.set_title(f'{data.patient_id}: '
                 + '  '.join(f'{s}: {n}' for (n, _, _, _), s
                             in zip(results, ('solid', 'dashed', 'dotted', 'dashdot'))))
    # save before show(), so the figure survives even if the window is closed or the
    # script runs without a display
    dvh_path = os.path.join(save_dir, 'dvh_imrt_comparison.png')
    plt.savefig(dvh_path, dpi=120, bbox_inches='tight')
    print(f'DVH saved: {dvh_path}')
    plt.show()


if __name__ == '__main__':
    console, log = sys.stdout, open(LOG_FILE, 'a')
    sys.stdout = sys.stderr = Tee(console, log)
    try:
        print(f'\n{"=" * 70}\n{time.strftime("%Y-%m-%d %H:%M:%S")}  '
              f'{NUM_BEAMS} beams x {APERTURES_PER_BEAM} apertures, '
              f'RUN_MIP={RUN_MIP}\n{"=" * 70}')
        imrt_global_optimal()
    finally:
        sys.stdout, sys.stderr = console, console
        log.close()
        print(f'log written to {os.path.abspath(LOG_FILE)}')