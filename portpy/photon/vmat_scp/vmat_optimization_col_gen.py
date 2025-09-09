# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use — including offering PortPy as a service,
# or incorporating it into a commercial product — requires
# a separate commercial license.
# ----------------------------------------------------------------------

from __future__ import annotations

from portpy.photon import Optimization
from typing import TYPE_CHECKING
import time
from portpy.photon.utils.get_apt_reg_metric import get_apt_reg_metric

if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
from portpy.photon.clinical_criteria import ClinicalCriteria
import cvxpy as cp
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
from scipy.spatial import cKDTree

from scipy.ndimage import binary_erosion, label
try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    pass
import json


class VmatOptimizationColGen(Optimization):
    def __init__(self, my_plan: Plan, arcs=None, inf_matrix: InfluenceMatrix = None,
                 clinical_criteria: ClinicalCriteria = None,
                 opt_params: dict = None, vars: dict = None, sol=None,
                 is_corr: bool = False, delta: np.ndarray = None):

        # combine steps objective functions into one
        if 'col_gen' in opt_params:
            super().__init__(my_plan=my_plan, inf_matrix=inf_matrix,
                             clinical_criteria=clinical_criteria,
                             opt_params=opt_params["col_gen"], vars=vars)
            opt_params = opt_params["col_gen"]  # col_gen is the new opt parameters dictionary for col gen
        else:

            # Call the constructor of the base class (Optimization) using super()
            super().__init__(my_plan=my_plan, inf_matrix=inf_matrix,
                             clinical_criteria=clinical_criteria,
                             opt_params=opt_params, vars=vars)
        # save previous solution if passed

        self.prev_sol = sol
        self.distance_from_structure = {}
        if arcs is None:
            self.arcs = my_plan.arcs
        else:
            self.arcs = arcs
        self.cvxpy_params = {}
        self.vmat_params = opt_params['opt_parameters']
        self.all_params = opt_params
        self.obj_funcs = None
        self.constraint_def = None
        self.outer_iteration = 0
        self.best_iteration = None
        self.obj_actual = []
        self.constraints_actual = []
        self.is_corr = is_corr
        self.constraints_ind_map = {}
        if delta is None:
            self.delta = np.zeros(self.inf_matrix.A.shape[0])

    def calculate_scores_grad_new(self, A, A_indices, sol):
        """
        Efficiently calculate scores for unselected beamlets using gradients.

        Parameters:
            A (csr_matrix): Influence matrix (voxels x beamlets).
            A_indices (list): List of indices of selected beamlets.
            sol (dict): Dictionary containing solution intensities ('y').

        Returns:
            list: List of tuples (beamlet index, score) sorted by score (lower is better).
        """
        # Step 1: Reconstruct B matrix from A_indices
        if not A_indices:  # Empty B
            self.B = np.empty((A.shape[0], 0))

        # Flatten A_indices and extract columns from A
        selected_beamlets = [item for sublist in A_indices for item in sublist]
        # Get unselected beamlets index
        selected_indices_set = set(selected_beamlets)
        unselected_mask = np.array([i not in selected_indices_set for i in range(A.shape[1])])

        # Step 2: Compute current dose with selected apertures
        current_dose = self.B @ sol['y'] if self.B.shape[1] > 0 else np.zeros(A.shape[0])

        # Initialize gradients
        num_beamlets = A.shape[1]
        overdose_grad = np.zeros(num_beamlets)
        underdose_grad = np.zeros(num_beamlets)
        quadratic_grad = np.zeros(num_beamlets)
        constraints_grad = np.zeros(num_beamlets)

        # Objective function gradients
        obj_funcs = self.obj_funcs
        structures = self.my_plan.structures
        inf_matrix = self.my_plan.inf_matrix
        num_fractions = self.my_plan.get_num_of_fractions()

        for obj in obj_funcs:
            if obj['type'] == 'quadratic-overdose':
                struct = obj['structure_name']
                if struct not in structures.get_structures():
                    continue
                voxels = inf_matrix.get_opt_voxels_idx(struct)
                if len(voxels) == 0:
                    continue
                dose_gy = self.get_dose_gy_per_frac_for_opt(obj, struct=struct)
                influence = A[voxels, :]
                overdose = current_dose[voxels] - dose_gy
                overdose[overdose < 0] = 0
                overdose_grad += (2 * obj['weight'] / len(voxels)) * (influence.T @ overdose)
            elif obj['type'] == 'quadratic-underdose':
                struct = obj['structure_name']
                if struct not in structures.get_structures():
                    continue
                voxels = inf_matrix.get_opt_voxels_idx(struct)
                if len(voxels) == 0:
                    continue
                dose_gy = self.get_dose_gy_per_frac_for_opt(obj, struct=struct)
                influence = A[voxels, :]
                underdose = current_dose[voxels] - dose_gy
                underdose[underdose > 0] = 0
                underdose_grad += (2 * obj['weight'] / len(voxels)) * (influence.T @ underdose)
            elif obj['type'] == 'quadratic':
                struct = obj['structure_name']
                if struct not in structures.get_structures():
                    continue
                voxels = inf_matrix.get_opt_voxels_idx(struct)
                if len(voxels) == 0:
                    continue
                influence = A[voxels, :]
                quadratic_grad += (2 * self.vmat_params['normalize_oar_weights'] * obj['weight'] / len(voxels)) * (influence.T @ current_dose[voxels])

        # Constraints gradients
        constraint_def = self.constraint_def
        constraints_ind_map = self.constraints_ind_map

        if constraints_ind_map:
            for constraint in constraint_def:

                if constraint['type'] == 'max_dose':
                    struct = constraint['parameters']['structure_name']
                    if struct not in structures.get_structures():
                        continue
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    if len(voxels) == 0:
                        continue
                    limit_key = self.matching_keys(constraint['constraints'], 'limit')
                    if limit_key:
                        limit = self.dose_to_gy(limit_key, constraint['constraints'][limit_key]) / num_fractions
                        dual = self.constraints[constraints_ind_map['max_dose_' + struct + str(limit)]].dual_value
                        constraints_grad += np.squeeze(np.asarray(dual * (A[voxels, :])))
                    # constraints_diff = (
                    #         np.sum(dual[:, None] * (current_dose[voxels, None] + delta*A[voxels, :] - limit / num_fractions), axis=0)
                    #         - np.sum(dual * (current_dose[voxels] - limit / num_fractions)))
                    # faster implemtation. Need to verify it
                    # perturbed_term = np.einsum('i,ij->j', dual, perturbed_doses[voxels] - limit / num_fractions)
                    # sol['constraints_grad'] += constraints_diff / delta
                if constraint['type'] == 'DFO':
                    dfo, oar_voxels = self.get_dfo_parameters(dfo_dict=constraint, is_obj=False)
                    dual = self.constraints[constraints_ind_map['max_dfo']].dual_value
                    constraints_grad += np.squeeze(np.asarray(dual * (A[oar_voxels, :])))

        # Combine all gradients to calculate final scores
        scores = (overdose_grad + underdose_grad + quadratic_grad + constraints_grad)
        scores = scores / np.max(abs(scores))
        # scores = scores/np.max(abs(scores[unselected_mask]))  # normalize only using unselected beamlets
        return scores

    def calculate_scores_fast(self, A, A_indices, sol, delta=1e-3):
        """
        Efficiently calculate scores for unselected beamlets using NumPy.

        Parameters:
            A (csr_matrix): Influence matrix (voxels x beamlets).
            A_indices (list): List of indices of selected beamlets.
            sol (dict): Dictionary containing solution intensities ('y').
            delta (float): Small perturbation value.

        Returns:
            list: List of tuples (beamlet index, score) sorted by score (lower is better).
        """
        # Step 1: Reconstruct B matrix from A_indices
        if not A_indices:  # Empty B
            self.B = np.empty((A.shape[0], 0))

        # Flatten A_indices and extract columns from A
        selected_beamlets = [item for sublist in A_indices for item in sublist]

        # Step 2: Compute current dose with selected apertures
        current_dose = self.B @ sol['y'] if self.B.shape[1] > 0 else np.zeros(A.shape[0])

        # Step 3: Combine selected indices into a set for fast exclusion
        selected_indices_set = set(selected_beamlets)

        # Step 4: Compute perturbed doses and scores in a vectorized manner
        unselected_mask = np.array([i not in selected_indices_set for i in range(A.shape[1])])
        unselected_beamlets = A[:, unselected_mask].toarray()  # Convert sparse matrix to dense
        perturbed_doses = current_dose[:, None] + delta * unselected_beamlets  # Broadcast addition

        # # Step 6: Compute scores for all unselected beamlets
        # pres_per_frac = self.my_plan.get_prescription() / self.my_plan.get_num_of_fractions()
        # ptv_vox = self.inf_matrix.get_opt_voxels_idx('PTV')
        # p = np.zeros(self.inf_matrix.A.shape[0])
        # p[ptv_vox] = pres_per_frac
        # diff_current = current_dose - p
        # diff_perturbed = perturbed_doses - p[:, None]
        # scores = (1 / A.shape[0]) * (
        #         np.sum(self.weights[:, None] * diff_perturbed ** 2, axis=0)
        #         - np.sum(self.weights * diff_current ** 2)
        # ) / delta
        obj_funcs = self.obj_funcs
        structures = self.my_plan.structures
        inf_matrix = self.my_plan.inf_matrix
        num_fractions = self.my_plan.get_num_of_fractions()
        sol['overdose_grad'] = 0
        sol['underdose_grad'] = 0
        sol['quadratic_grad'] = 0
        sol['constraints_grad'] = 0
        obj_ind = 0
        # check if we have smooth objective
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(
                            struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    dose_key = self.matching_keys(obj_funcs[i], 'dose_')
                    dose_gy = self.dose_to_gy(dose_key, obj_funcs[i][dose_key]) / num_fractions
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    # new_obj_value = (1 / len(voxels)) * obj_funcs[i]['weight'] * np.sum((np.maximum(0, (new_dose[voxels] - dose_gy)) ** 2))
                    # old_obj_value = (1 / len(voxels)) * obj_funcs[i]['weight'] * np.sum((np.maximum(0, (current_dose[voxels] - dose_gy)) ** 2))
                    obj_diff = (1 / len(voxels)) * (
                                np.sum(obj_funcs[i]['weight'] * np.maximum(0, (perturbed_doses[voxels] - dose_gy)) ** 2, axis=0)
                                - np.sum(obj_funcs[i]['weight'] * np.maximum(0, (current_dose[voxels] - dose_gy)) ** 2))
                    sol['overdose_grad'] += obj_diff / delta
                    obj_ind = obj_ind + 1
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(struct)) == 0:
                        continue
                    dose_key = self.matching_keys(obj_funcs[i], 'dose_')
                    dose_gy = self.dose_to_gy(dose_key, obj_funcs[i][dose_key]) / num_fractions
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    # new_obj_value = (1 / len(voxels) * obj_funcs[i]['weight'] * np.sum(np.maximum(0, (dose_gy - new_dose[voxels])) ** 2))
                    # old_obj_value = (1 / len(voxels) * obj_funcs[i]['weight'] * np.sum(np.maximum(0, (dose_gy - current_dose[voxels])) ** 2))
                    # sol['underdose_grad'] += (new_obj_value - old_obj_value) / delta
                    obj_diff = (1 / len(voxels)) * (
                            np.sum(obj_funcs[i]['weight'] * np.maximum(0, (dose_gy - perturbed_doses[voxels])) ** 2, axis=0)
                            - np.sum(obj_funcs[i]['weight'] * np.maximum(0, (dose_gy - current_dose[voxels])) ** 2))
                    sol['underdose_grad'] += obj_diff / delta
                    obj_ind = obj_ind + 1
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(struct)) == 0:
                        continue
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    # new_obj_value = (1 / len(voxels)) * obj_funcs[i]['weight'] * np.sum((new_dose[voxels] ** 2))
                    # old_obj_value = (1 / len(voxels)) * obj_funcs[i]['weight'] * np.sum((current_dose[voxels] ** 2))
                    # sol['quadratic_grad'] += (new_obj_value - old_obj_value) / delta
                    obj_diff = (1 / len(voxels)) * (
                            np.sum(obj_funcs[i]['weight'] * (perturbed_doses[voxels] ** 2), axis=0)
                            - np.sum(obj_funcs[i]['weight'] * (current_dose[voxels] ** 2)))
                    sol['quadratic_grad'] += obj_diff / delta
                    obj_ind = obj_ind + 1

        # constraints
        constraint_def = self.constraint_def
        constraints_ind_map = self.constraints_ind_map
        if constraints_ind_map:
            # extract dual values of the constraints
            for i in range(len(constraint_def)):
                if constraint_def[i]['type'] == 'max_dose':
                    limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                    if limit_key:
                        limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                        org = constraint_def[i]['parameters']['structure_name']
                        if org in self.my_plan.structures.get_structures():
                            if len(inf_matrix.get_opt_voxels_idx(org)) == 0:
                                continue
                            voxels = inf_matrix.get_opt_voxels_idx(org)
                            if len(self.constraints) > 0:
                                dual = self.constraints[constraints_ind_map['max_dose_' + org + str(limit / num_fractions)]].dual_value
                                constraints_diff = (
                                        np.sum(dual[:, None] * (perturbed_doses[voxels] - limit / num_fractions), axis=0)
                                        - np.sum(dual * (current_dose[voxels] - limit / num_fractions)))
                                # faster implemtation. Need to verify it
                                # perturbed_term = np.einsum('i,ij->j', dual, perturbed_doses[voxels] - limit / num_fractions)
                                sol['constraints_grad'] += constraints_diff / delta
        scores = sol['overdose_grad'] + sol['underdose_grad'] + sol['quadratic_grad'] + sol['constraints_grad']
        # scores.append((i, score))
        # Step 7: Map scores to beamlet indices
        unselected_indices = np.where(unselected_mask)[0]
        scores_list = list(zip(unselected_indices, scores))

        return scores_list

    def select_best_aperture_smooth_greedy(self, remaining_beam_ids: list, scores, smooth_delta: float = 0.1, smooth_beta: float = 5, apt_reg_type_cg: str = 'row-by-row'):
        """
        This function helps to find the aperture with best score.
        It solves pricing problem to find the best configuration of leaf positions for given scores

        :param remaining_beam_ids: List of Remaining beams ids for which score needs to be calculated
        :param scores: score vector for all the beamlets
        :param smooth_delta: parameter to regularize the aperture leaf
        :param smooth_beta: parameter for exponential smoothing between leafs. it is not used during row-by-row
        :param apt_reg_type_cg: either 'row-by-row' or 'greedy'. row by row smooth leaf sequentially.
                But greedy choose the next best leaf for smoothness
        :return: aperture with best score(s)
        """

        aperture_score = {}
        selected_beamlets = {}
        selected_leafs = {}
        # Map beamlet indices to their scores for quick lookup
        if isinstance(scores, np.ndarray):
            scores = list(scores)
            score_dict = {index: score for index, score in enumerate(scores)}
        else:
            score_dict = {index: score for index, score in scores}

        # Count how many control points have been selected from each arc
        arc_selection_count = {arc['arc_id']: arc['num_beams'] for arc in self.arcs.arcs_dict['arcs']}
        # score_dict_orig = score_dict.copy()
        for arc in self.arcs.arcs_dict['arcs']:
            for beam in arc['vmat_opt']:
                if beam['beam_id'] in remaining_beam_ids:
                    # Update count of selected control points for this arc
                    arc_selection_count[arc['arc_id']] -= 1
                    cumulative_score = 0
                    selected_beamlets[beam['beam_id']] = []
                    selected_leafs[beam['beam_id']] = [[] for _ in range(beam['num_rows'])]
                    remaining_rows = list(range(beam['num_rows']))  # Start with all rows
                    processed_rows = []
                    r_best = -1
                    while remaining_rows:

                        # identify the best row
                        # get scores for each remaining row

                        if apt_reg_type_cg == 'greedy':
                            row_scores = [sum(score_dict[beamlet] for beamlet in row[row >= 0]) for r, row in enumerate(beam['reduced_2d_grid']) if r in remaining_rows]
                            r_best = np.argmin(row_scores)
                            row = beam['reduced_2d_grid'][remaining_rows[r_best]]
                        else:
                            r_best += 1
                            row = beam['reduced_2d_grid'][r_best]


                        # scores_matrix = np.vectorize(lambda x: float(score_dict.get(x, 0)))(beam['reduced_2d_grid'])
                        # Step 0: Initialization
                        c1, c2 = np.argwhere(row >= 0)[0][0], np.argwhere(row >= 0)[0][0]-1 # start with one index less for c2
                        c1_opt, c2_opt = c1, c2 + 1
                        v, v_bar, v_star = 0, 0, 0  # Initialize v, v_bar, and v_star

                        # End beamlet index in the row
                        n = np.argwhere(row >= 0)[-1][0]

                        # Step 1: Iterate over possible right leaf positions
                        while c2 < n:
                            # Calculate v incrementally by adding beamlet scores in the range [c1, c2]
                            v += score_dict.get(row[c2 + 1])

                            # Step 2: Update v_bar if necessary
                            if v > v_bar:
                                v_bar = v
                                c1 = c2+2  # Update left position

                            elif v - v_bar < v_star:  # Update v_star if the reduced cost improves
                                v_star = v - v_bar
                                c1_opt, c2_opt = c1, c2 + 1  # Update optimal aperture

                            # Step 3: Increment c2
                            c2 += 1

                        # Return the optimal leaf positions, reduced cost, and beamlets in the aperture
                        optimal_beamlets = row[c1_opt:c2_opt + 1]
                        # left, right, row_score, open_beamlet_indices = self.find_best_leaf_position_fast(row, scores)
                        # left, right, row_score, open_beamlet_indices = self.find_best_leaf_position_fast(row, scores)
                        cumulative_score += v_star
                        selected_beamlets[beam['beam_id']].extend(optimal_beamlets)

                        # remove the best row from configuration
                        if apt_reg_type_cg == 'greedy':
                            # # change back scores of the unselected beamlets to original after finding best row
                            # for beamlet in score_dict:
                            #     if beamlet not in selected_beamlets[beam['beam_id']]:
                            #         score_dict[beamlet] = score_dict_orig[beamlet]
                            selected_leafs[beam['beam_id']][remaining_rows[r_best]] = [c1_opt - 1, c2_opt + 1] # vmat scp format
                            processed_rows.append(remaining_rows[r_best])
                            del remaining_rows[r_best]
                        else:
                            selected_leafs[beam['beam_id']][r_best] = [c1_opt - 1, c2_opt + 1]
                            processed_rows.append(remaining_rows[0])
                            del remaining_rows[0]

                        # update the score of other rows beamlets based on smoothness penalty with discount factor
                        # Step 2: Penalize remaining rows based on the best row
                        if apt_reg_type_cg == 'greedy':
                            prev_c1_opt = selected_leafs[beam['beam_id']][processed_rows[-1]][0] + 1
                            prev_c2_opt = selected_leafs[beam['beam_id']][processed_rows[-1]][1] - 1
                            for r in remaining_rows:
                                row = beam['reduced_2d_grid'][r]
                                row_distance = abs(r - processed_rows[-1]) - 1  # subtracted one to keep row discount exp(0)=1 for the nearest row
                                row_discount = np.exp(-smooth_beta * row_distance)  # Exponential discount based on row distance
                                # Define the boundary window size. It is half of BEV length of open aperture row

                                for c, beamlet in enumerate(row):
                                    if beamlet >= 0:
                                        # left beamlets
                                        if c < prev_c1_opt:  # left of previous left
                                            score_dict[beamlet] += smooth_delta * row_discount * abs(prev_c1_opt - c)  # decrease score to close the beamlet
                                        elif c > prev_c2_opt:  # right of previous right
                                            score_dict[beamlet] += smooth_delta * row_discount * abs(c - prev_c2_opt)  # decrease score
                                        elif prev_c1_opt < c < prev_c2_opt:  # right of previous left
                                            if (c - prev_c1_opt) < (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1*smooth_delta * row_discount * abs(prev_c1_opt - c)  # increase score (-ve of distance)
                                            if (c - prev_c1_opt) > (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1*smooth_delta * row_discount * abs(c - prev_c2_opt)  # increase score (-ve of distance)

                            # beam['scores_matrix'] = np.vectorize(lambda x: float(score_dict.get(x, 0)))(beam['reduced_2d_grid'])
                        else:
                            # Add penalty for deviations from the upper row
                            # Define the boundary window size. It is half of BEV length of open aperture row
                            if r_best < beam['num_rows']-1:
                                row = beam['reduced_2d_grid'][r_best + 1]
                                prev_c1_opt = selected_leafs[beam['beam_id']][r_best][0] + 1
                                prev_c2_opt = selected_leafs[beam['beam_id']][r_best][1] - 1
                                for c, beamlet in enumerate(row):
                                    if beamlet >= 0:
                                        # left beamlets
                                        if c < prev_c1_opt:  # left of previous left
                                            score_dict[beamlet] += smooth_delta * abs(prev_c1_opt - c)  # decrease score to close the beamlet
                                        elif c > prev_c2_opt:  # right of previous right
                                            score_dict[beamlet] += smooth_delta * abs(c - prev_c2_opt)  # decrease score
                                        elif prev_c1_opt < c < prev_c2_opt:  # right of previous left
                                            if (c - prev_c1_opt) < (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1*smooth_delta * abs(prev_c1_opt - c)  # increase score (-ve of distance)
                                            if (c - prev_c1_opt) > (prev_c2_opt - c):  # -ve penalize only beamlets if they are in optimal configuration
                                                score_dict[beamlet] += -1*smooth_delta * abs(c - prev_c2_opt)  # increase score (-ve of distance)
                        aperture_score[beam['beam_id']] = cumulative_score
        # best beam id
        # best_beam_id = min(aperture_score, key=aperture_score.get)
        if 'update_balanced_arc_score' in self.vmat_params:
            if self.vmat_params['update_balanced_arc_score']:
                # Update scores based on arc balance factor
                if 'update_balanced_arc_score_p' in self.vmat_params:
                    p  = self.vmat_params['update_balanced_arc_score_p']
                else:
                    p = -1
                # epsilon = 1e-3  # Small regularization term to avoid zero contribution
                # M = max(arc_selection_count.values()) + epsilon # added epsilon to handle arcs with 0
                total_arcs = len(arc_selection_count)
                if total_arcs > 1:
                    # # Ensure all arcs have at least one selected beam when only one arc has a selection
                    # if sum(count > 0 for count in arc_selection_count.values()) == 1:
                    #     for arc_id in arc_selection_count.keys():
                    #         if arc_selection_count[arc_id] == 0:
                    #             arc_selection_count[arc_id] = 1  # Add a dummy beam count to other arcs
                    arc_scores = {}
                    arc_weights = {arc_id: (1+count) ** (1 / p) for arc_id, count in
                                   arc_selection_count.items()}  # Compute m^(1/p) for each arc
                    sum_arc_weights = sum(arc_weights.values())  # Sum of all arc weights


                    for arc_id in arc_selection_count.keys():
                        arc_scores[arc_id] = arc_weights[arc_id] / sum_arc_weights if sum_arc_weights > 0 else 0.5 # lower arc count should get higher weight
                    print(arc_scores)

                    # Apply arc-based score adjustment
                    for arc in self.arcs.arcs_dict['arcs']:
                        for beam in arc['vmat_opt']:
                            if beam['beam_id'] in remaining_beam_ids:
                                for r in range(beam['num_rows']):
                                    row = beam['reduced_2d_grid'][r]
                                    for beamlet in row:
                                        if beamlet >= 0:
                                            score_dict[beamlet] *= arc_scores[arc['arc_id']] # update scores of each remaining beamlet
                                aperture_score[beam['beam_id']] = aperture_score[beam['beam_id']] * arc_scores[arc['arc_id']] # update cummulative score

        if 'num_select_apertures' in self.vmat_params:
            num_select_apertures = self.vmat_params['num_select_apertures']
        else:
            num_select_apertures = 1
        top_beam_ids = sorted(aperture_score, key=aperture_score.get)[:num_select_apertures]
        # print('Best reduced cost is of beam id ', best_beam_id, ':', str(aperture_score[best_beam_id]))
        for i, beam_id in enumerate(top_beam_ids, 1):
            print(f'#{i} best reduced cost is of beam ID {beam_id}: {aperture_score[beam_id]}')
        # save leaf positions for the best found beam to arcs dictionary for vmat scp optimization
        for arc in self.arcs.arcs_dict['arcs']:
            for beam in arc['vmat_opt']:
                if beam['beam_id'] in top_beam_ids:
                    beam['leaf_pos_cg'] = selected_leafs[beam['beam_id']]
                    # update leaf positions in arcs dictionary to be used for vmat scp
                    beam['leaf_pos_f'] = selected_leafs[beam['beam_id']]
                    beam['leaf_pos_b'] = selected_leafs[beam['beam_id']]
                    beam['leaf_pos_left'] = [selected_leafs[beam['beam_id']][i][0] for i in range(len(selected_leafs[beam['beam_id']]))]
                    beam['leaf_pos_right'] = [selected_leafs[beam['beam_id']][i][1] for i in range(len(selected_leafs[beam['beam_id']]))]
                    # beam['apt_sol_gen'] = np.isin(beam['reduced_2d_grid'], selected_beamlets[beam['beam_id']]).astype(int)
                    # beam['scores_matrix'] = np.vectorize(lambda x: float(score_dict.get(x, 0)))(beam['reduced_2d_grid'])
                if beam['beam_id'] in remaining_beam_ids:
                    if 'output_score_matrix' in self.vmat_params:
                        if self.vmat_params['output_score_matrix']:
                            beam['apt_sol_gen'] = np.isin(beam['reduced_2d_grid'], selected_beamlets[beam['beam_id']]).astype(int)
                            beam['scores_matrix'] = np.vectorize(lambda x: float(score_dict.get(x, 0)))(beam['reduced_2d_grid'])
                    beam['aperture_score'] = aperture_score[beam['beam_id']]
        return top_beam_ids, [selected_beamlets[b_id] for b_id in top_beam_ids]

    def solve_rmp(self, *args, **kwargs):
        """
            Solve restricted master problem to get optimal aperture intensity

        :param args: arguments for cvxpy problem
        :param kwargs: keywords for cvxpy problem
        :return: solution for the restricted master problem
        """
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        opt_params = self.opt_params
        clinical_criteria = self.clinical_criteria
        # Create cvxpy problem for RMP
        # get opt params for optimization
        obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
        A = inf_matrix.A
        B = self.B
        num_fractions = clinical_criteria.get_num_of_fractions()
        ptv_vox = inf_matrix.get_opt_voxels_idx('PTV')
        p = np.zeros((A.shape[0]))
        p[ptv_vox] = clinical_criteria.get_prescription() / num_fractions

        if np.all(self.B == 0):
            return {'y': np.zeros(self.B.shape[1]), 'obj_value':0}

        st = inf_matrix
        y = cp.Variable(self.B.shape[1], pos=True)
        print('Objective Start')
        self.obj = []
        self.constraints = []
        obj = self.obj
        constraints = self.constraints


        # obj += [(1 / B.shape[0]) * cp.sum_squares(cp.multiply(np.sqrt(self.weights), B @ y - p))]
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    dose_gy = self.get_dose_gy_per_frac_for_opt(obj_funcs[i], struct=struct)
                    dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dO))]
                    constraints += [B[st.get_opt_voxels_idx(struct), :] @ y <= dose_gy + dO]
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    dose_gy = self.get_dose_gy_per_frac_for_opt(obj_funcs[i], struct=struct)
                    dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dU))]
                    constraints += [B[st.get_opt_voxels_idx(struct), :] @ y >= dose_gy - dU]
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    obj += [self.vmat_params['normalize_oar_weights'] * (1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(B[st.get_opt_voxels_idx(struct), :] @ y))]
        constraint_def = self.constraint_def
        self.constraints_ind_map = {}  # empty previous values if any
        constraints_ind_map = self.constraints_ind_map
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    if org in my_plan.structures.get_structures():
                        if len(st.get_opt_voxels_idx(org)) == 0:
                            continue
                        constraints += [B[st.get_opt_voxels_idx(org), :] @ y <= limit / num_fractions]
                        constraints_ind_map['max_dose_' + org + str(limit / num_fractions)] = len(constraints) - 1
            # elif constraint_def[i]['type'] == 'mean_dose':
            #     limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
            #     if limit_key:
            #         limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
            #         org = constraint_def[i]['parameters']['structure_name']
            #         # mean constraints using voxel weights
            #         if org in my_plan.structures.get_structures():
            #             if len(st.get_opt_voxels_idx(org)) == 0:
            #                 continue
            #             fraction_of_vol_in_calc_box = my_plan.structures.get_fraction_of_vol_in_calc_box(org)
            #             limit = limit/fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
            #             constraints += [(1 / sum(st.get_opt_voxels_volume_cc(org))) *
            #                             (cp.sum((cp.multiply(st.get_opt_voxels_volume_cc(org),
            #                                                  A[st.get_opt_voxels_idx(org), :] @ x))))
            #                             <= limit / num_fractions]
            elif constraint_def[i]['type'] == 'DFO':
                dfo, oar_voxels = self.get_dfo_parameters(dfo_dict=constraint_def[i], is_obj=False)
                constraints += [B[oar_voxels, :] @ y <= dfo / num_fractions]
                constraints_ind_map['max_dfo'] = len(constraints) - 1
        problem = cp.Problem(cp.Minimize(cp.sum(self.obj)), constraints=self.constraints)

        print('Running Optimization..')
        t = time.time()
        problem.solve(*args, **kwargs)
        sol = {'y': y.value if y.shape[0] != 1 else np.array(y.value), 'obj_value': problem.value}
        elapsed = time.time() - t
        print("Elapsed time: {}".format(elapsed))
        print("Optimal value: %s" % problem.value)

        return sol

    def run_col_gen_algo(self, *args, **kwargs):
        """
            VMAT Optimization using Column Generation.

            This module implements the column generation approach for directly optimizing
            MLC (multi-leaf collimator) leaf positions in Volumetric Modulated Arc Therapy (VMAT).

            Column generation is a powerful greedy-type algorithm used in operations research and
            has been successfully adapted for radiotherapy optimization. Instead of optimizing
            beamlet intensities first and then performing leaf sequencing, this approach
            iteratively selects the most promising apertures (beam shapes) and adds them to a
            pool of candidate apertures.

            The algorithm proceeds as follows:

            1. Start with an initial set of apertures indexed by k = 1, ..., K.
            2. Construct the dose-influence matrix B using columns B_k corresponding to each aperture.
            3. Solve the optimization problem:

                   minimize     f(d = By)
                   subject to   g(d) <= 0

               where `y` is the aperture intensity vector, `f` is the objective function, and
               `g` represents constraints (e.g., dose limits).

            4. Using the optimal dual variables λ (from CVXPY), construct the Lagrangian:

                   L(d, λ) = f(d) + λᵀ g(d)

            5. Compute the gradient of the Lagrangian w.r.t. beamlet intensities x:

                   ∂L/∂x = Aᵀ ∂L/∂d

               where A is the original beamlet-based dose-influence matrix.

            6. Each beamlet receives a score representing the estimated improvement to the
               objective if activated. Using this, identify the best aperture per beam by
               minimizing the cumulative score of open beamlets.

            7. Append the selected aperture to the set and repeat until convergence or stopping criteria.

            Note:
            - This approach is greedy and does not guarantee global optimality.
            - However, it yields high-quality solutions that can serve as warm starts
              for more sophisticated methods such as Sequential Convex Programming (SCP).

            References:
            - Preciado-Walters et al., 2004
            - Romeijn et al., 2005

        :param args: arguments for cvxpy problem
        :param kwargs: keywords for cvxpy problem

        Returns:
            sol (dict): Solution dictionary including final beamlet intensities,
                        aperture selection, and smoothness metrics.
        """

        A_indices = []  # List of selected beamlet index sets for B columns
        self.arcs.get_initial_leaf_pos(initial_leaf_pos='BEV')

        # Evaluate initial aperture regularization metric
        apt_reg_obj = get_apt_reg_metric(my_plan=self.my_plan)
        print(f'#### Initial apt_reg_obj (BEV): {apt_reg_obj} ####')

        arcs = self.arcs.arcs_dict['arcs']
        A = self.inf_matrix.A
        self.B = np.empty((A.shape[0], 0))  # Start with empty B matrix (no selected apertures)

        # Flatten all beam IDs across arcs
        remaining_beam_ids = [beam['beam_id'] for arc in arcs for beam in arc['vmat_opt']]
        map_beam_id_to_index = {
            beam_id: index
            for index, beam_id in enumerate(
                beam_id for arc in arcs for beam_id in arc['beam_ids']
            )
        }

        B_indices = []
        sol = {'y': np.zeros(0)}  # Initial RMP solution

        # Objective functions and constraints setup
        self.obj_funcs = self.opt_params.get('objective_functions', [])
        opt_params_constraints = self.opt_params.get('constraints', [])
        constraint_def = deepcopy(self.clinical_criteria.get_criteria())

        # Update clinical criteria with optimization-specific constraints
        for opt_constraint in opt_params_constraints:
            # add constraint
            param = opt_constraint['parameters']
            if param['structure_name'] in self.my_plan.structures.get_structures():
                criterion_exist, criterion_ind = self.clinical_criteria.check_criterion_exists(opt_constraint, return_ind=True)
                if criterion_exist:
                    constraint_def[criterion_ind] = opt_constraint
                else:
                    constraint_def += [opt_constraint]
        self.constraint_def = constraint_def

        # Smoothing and regularization parameters
        smooth_delta = self.vmat_params.get('smooth_delta', 0)
        smooth_beta = self.vmat_params.get('smooth_beta', 1)
        apt_reg_type_cg = self.vmat_params.get('apt_reg_type_cg', 'row-by-row')
        self.vmat_params.setdefault('normalize_oar_weights', 1)

        convergence = []

        # Main column generation loop
        while remaining_beam_ids:
            # Compute gradient-based scores for selecting promising apertures
            scores = self.calculate_scores_grad_new(A, A_indices, sol)

            # Select top aperture candidates based on scores
            top_beam_ids, beam_beamlets_list = self.select_best_aperture_smooth_greedy(
                remaining_beam_ids,
                scores,
                smooth_delta=smooth_delta,
                smooth_beta=smooth_beta,
                apt_reg_type_cg=apt_reg_type_cg
            )

            # Add selected apertures to B matrix
            for beam_id, open_beamlet_indices in zip(top_beam_ids, beam_beamlets_list):
                A_indices.append(open_beamlet_indices)
                B_indices.append(map_beam_id_to_index[beam_id])
                # Sum beamlets for this beam ID and add as new column to B matrix
                self.B = np.column_stack([self.B, A[:, open_beamlet_indices].sum(axis=1).A1])
            # Solve RMP
            sol = self.solve_rmp(*args, **kwargs)  # Use only selected beamlets
            # # save statistics to convergence list
            convergence.append([top_beam_ids[0],
                                get_apt_reg_metric(self.my_plan, beam_ids=top_beam_ids),
                                sol['obj_value']])
            # Remove used control point
            # remaining_beam_ids.remove(best_beam_id)
            remaining_beam_ids = list(set(remaining_beam_ids) - set(top_beam_ids))

        # Recover full beamlet intensity vector from RMP solution
        x = np.zeros(A.shape[1])
        for b in range(len(B_indices)):
            x[A_indices[b]] = sol['y'][b]
        sol['optimal_intensity'] = x
        # cache these leaf positions to solution
        sol['arcs'] = self.arcs.arcs_dict['arcs']
        # Col Gen Convergence
        import pandas as pd
        df = pd.DataFrame(convergence)
        print(df.to_string())

        # get smoothness function value using leaf positions in arcs
        apt_reg_obj = get_apt_reg_metric(my_plan=self.my_plan)
        print('####apr_reg_obj after col gen: {} ####'.format(apt_reg_obj))
        sol['apt_reg_obj'] = apt_reg_obj

        return sol

    def get_dfo_interior(self, struct_name: str = 'GTV', min_dose: float = None, max_dose: float = None, pres: float = None):

        # get boundary and calc distance for interior voxels
        voxels = self.inf_matrix.get_opt_voxels_idx(struct_name)
        if min_dose is not None and max_dose is not None:
            if 'dfo_target_interior' not in self.clinical_criteria.clinical_criteria_dict:
                self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'] = {}
            if struct_name not in self.clinical_criteria.clinical_criteria_dict['dfo_target_interior']:
                # Assuming `target_mask` is your 3D binary mask with 1s inside all target structures
                target_mask = self.my_plan.structures.get_structure_mask_3d(struct_name)

                # Step 1: Label each sub-region in the mask
                labeled_mask, num_regions = label(target_mask)

                # Set up parameters for distance computation
                voxel_resolution = np.array(self.inf_matrix.opt_voxels_dict['ct_voxel_resolution_xyz_mm'][::-1])
                ct_origin = np.array(self.inf_matrix.opt_voxels_dict['ct_origin_xyz_mm'][::-1])

                # Get all GTV voxel coordinates in physical space
                vox_coord_xyz_mm = self.inf_matrix.opt_voxels_dict['voxel_coordinate_XYZ_mm'][0]
                interior_points = vox_coord_xyz_mm[voxels, :]  # All GTV voxel coordinates

                # List to accumulate boundary coordinates from each region
                all_boundary_coords = []

                for region_id in range(1, num_regions + 1):
                    # Extract the mask for the current region
                    region_mask = (labeled_mask == region_id)

                    # Identify boundary voxels for this region
                    eroded_region = binary_erosion(region_mask)
                    boundary_mask = region_mask & ~eroded_region
                    boundary_voxels = np.argwhere(boundary_mask)

                    # Convert boundary voxels to physical coordinates
                    boundary_coords = boundary_voxels * voxel_resolution + ct_origin
                    boundary_coords = boundary_coords[:, [2, 1, 0]]  # Convert ZYX to XYZ

                    # Accumulate boundary coordinates for this region
                    all_boundary_coords.append(boundary_coords)

                # Combine all boundary coordinates into a single array
                all_boundary_coords = np.vstack(all_boundary_coords)

                # Step 4: Use Nearest Neighbors to find the distance from each interior point to the nearest boundary point
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(all_boundary_coords)
                distances_voxels, _ = nbrs.kneighbors(interior_points)

                def fit_exponential_growth(x1, y1, x2, y2):
                    b = (np.log(y2) - np.log(y1)) / (x2 - x1)
                    a = y1 / np.exp(b * x1)
                    return a, b

                # Calculate a and b based on the given points
                a, b = fit_exponential_growth(np.min(distances_voxels), min_dose, np.max(distances_voxels), max_dose)

                prescription = np.squeeze(a * np.exp(b * distances_voxels))
                self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'][struct_name] = prescription
                # self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'] = {struct_name + 'distance_from_boundary_mm': distances_voxels}
            else:
                prescription = self.clinical_criteria.clinical_criteria_dict['dfo_target_interior'][struct_name]
        else:
            prescription = np.repeat(pres, len(voxels))
        return prescription

    def get_dose_gy_per_frac_for_opt(self, obj, struct):
        num_fractions = self.my_plan.get_num_of_fractions()
        dose_key = self.matching_keys(obj, 'dose_')
        dosemin_key = self.matching_keys(obj, 'dosemin')
        dosemax_key = self.matching_keys(obj, 'dosemax')
        dose_gy = None
        if dose_key:
            dose_gy = self.dose_to_gy(dose_key, obj[dose_key])/num_fractions
        elif dosemax_key:
            dosemin_gy = self.dose_to_gy(dosemin_key, obj[dosemin_key])/num_fractions
            dosemax_gy = self.dose_to_gy(dosemax_key, obj[dosemax_key])/num_fractions
            dose_gy = self.get_dfo_interior(struct_name=struct, min_dose=dosemin_gy, max_dose=dosemax_gy)
        return dose_gy

    def get_dfo_parameters(self, dfo_dict, is_obj: bool = False):
        weight_interpolate = None
        if not is_obj:
            param = dfo_dict['parameters']
            struct_name = param['structure_name']
            key = self.matching_keys(dfo_dict['constraints'], 'dose')
            max_dose = np.asarray([self.dose_to_gy(key, dose) for dose in dfo_dict['constraints'][key]])
            distance = np.asarray(param['distance_from_structure_mm'])
        else:
            struct_name = dfo_dict['structure_name']
            distance = np.asarray(dfo_dict['distance_from_structure_mm'])
            key = self.matching_keys(dfo_dict, 'dose')
            max_dose = np.asarray([self.dose_to_gy(key, dose) for dose in dfo_dict[key]])
            weight = np.asarray(dfo_dict['weight'])
            weight_interpolate = interp1d(distance, weight, kind='next')
        dfo_interpolate = interp1d(distance, max_dose, kind='next')
        target_voxels = self.inf_matrix.get_opt_voxels_idx(struct_name)
        all_vox = self.my_plan.inf_matrix.get_opt_voxels_idx('BODY')
        oar_voxels = np.setdiff1d(all_vox, target_voxels)
        vox_coord_xyz_mm = self.inf_matrix.opt_voxels_dict['voxel_coordinate_XYZ_mm'][0]

        if 'distance_from_structure_mm' not in self.inf_matrix.opt_voxels_dict:
            self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'] = {}
        if struct_name not in self.inf_matrix.opt_voxels_dict['distance_from_structure_mm']:
            print('calculating distance of normal tissue voxels from target for DFO constraints. This step may take some time..')
            start = time.time()
            dist_from_structure, _ = cKDTree(vox_coord_xyz_mm[target_voxels, :]).query(vox_coord_xyz_mm[oar_voxels, :], 1)
            # a = spatial.distance.cdist(, vox_coord_xyz_mm[PTV, :]).min(axis=1)
            print('Time for calc distance {}'.format(time.time() - start))
            # dist_from_structure = np.zeros_like(all_vox, dtype=float)
            # dist_from_structure[oar_voxels] = a
            self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'][struct_name] = dist_from_structure
        if not is_obj:
            return dfo_interpolate(self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'][struct_name]), oar_voxels
        else:
            return dfo_interpolate(self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'][struct_name]), weight_interpolate(self.inf_matrix.opt_voxels_dict['distance_from_structure_mm'][struct_name]), oar_voxels

    def remove_duplicates(self, dict_list):
        seen = set()
        unique_dicts = []

        for d in dict_list:
            key_str = json.dumps(d, sort_keys=True)  # Convert dict to JSON string (sorted)
            if key_str not in seen:
                seen.add(key_str)
                unique_dicts.append(d)

        return unique_dicts