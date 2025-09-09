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

import numpy as np
from portpy.photon.data_explorer import DataExplorer
from portpy.photon.influence_matrix import InfluenceMatrix
import json
from copy import deepcopy
from typing import Union, List


class Arcs:
    """
    A class representing beams_dict.

    - **Attributes** ::

        :param arcs_dict: arcs_dict dictionary that contains information about arcs in the format of dict
        dict: {'arcs':
                   'ID': list(int),
                   'gantry_angle': list(float),
                   'collimator_angle': list(float) }
                  }

        :type arcs_dict: dict
        :param inf_matrix: object of InfluenceMatrix class
        :type inf_matrix: object
        :param file_name: json file containing arcs information
        :type file_name: str
        :param data: object of DataExplorer class
        :type data: object


    - **Methods** ::

        method get_all_arcs: Get all arcs as a list
        method get_arc: Get arc based upon arc id
        method get_initial_leaf_pos: Get initial leaf position based upon BEV or other user defined criteria to start SCP
        method gen_interior_and_boundary_beamlets: Generate interior and boundary beamlets based upon step size
        method calc_actual_from_intermediate_sol: Calculate actual solution from intermediate solution



    """

    def __init__(self, inf_matrix: InfluenceMatrix, file_name: str = None, data: DataExplorer = None, arcs_dict: dict = None):
        """

        :param file_name: json file containing arcs information
        :data: object of DataExplorer class
        :arcs_dict: dictionary containing arcs information

        """
        if file_name is not None:
            self.arcs_dict = self.load_json(file_name)
        if arcs_dict is not None:
            self.arcs_dict = arcs_dict
        if data is not None:
            metadata = data.load_metadata()
            self.arcs_dict = metadata['arcs']
        self._inf_matrix = inf_matrix
        self.preprocess()

    def load_json(self, arcs_json_file):
        # store in arcs dictionary
        f = open(arcs_json_file)
        arcs_dict = json.load(f)
        f.close()
        return arcs_dict

    def get_all_arcs(self):
        """
        Get all arcs as a list

        """
        return self.arcs_dict['arcs']

    def get_arc(self, arc_id: Union[Union[int, str], List[Union[int, str]]]):
        """
        Get arc based upon arc id
        :param arc_id: arc id for the arc needed
        :return: list of arcs

        """
        ind = []
        if isinstance(arc_id, int) or isinstance(arc_id, str):
            ind = [i for i in range(len(self.arcs_dict['arcs'])) if
                   self.arcs_dict['arcs'][i]['arc_id'] == arc_id]
            if not ind:
                raise ValueError("invalid arc id {}".format(arc_id))
        arcs = []
        for a in ind:
            arc = self.arcs_dict['arcs'][a]
            arcs.append(arc)
        if isinstance(arc_id, int) or isinstance(arc_id, str):
            return  arcs[0]
        else:
            return arcs

    def get_max_cols(self):
        max_cols = 0
        arcs = self.arcs_dict['arcs']
        for a, arc in enumerate(arcs):
            for b, beam in enumerate(arc['vmat_opt']):
                max_cols = np.maximum(beam['reduced_2d_grid'].shape[1], max_cols)
        return max_cols

    def preprocess(self):
        arcs_dict = self.arcs_dict
        inf_matrix = self._inf_matrix
        all_cp_ids = inf_matrix.get_all_beam_ids()

        # get beams for each arc
        for i, arc in enumerate((arcs_dict['arcs'])):
            cp_ids = arc["beam_ids"]
            ind_access = [all_cp_ids.index(cp_id) for cp_id in cp_ids]
            beams_list = [deepcopy(inf_matrix.beamlets_dict[ind]) for ind in ind_access]
            arc['vmat_opt'] = beams_list

        # Precompute values for all arcs
        for arc in arcs_dict['arcs']:
            beams_list = arc['vmat_opt']

            # Precompute values for all beams in the arc
            start_beamlet_idxs = []
            end_beamlet_idxs = []
            num_rows_list = []
            num_cols_list = []
            start_leaf_pairs = []
            end_leaf_pairs = []
            min_positions_x = []
            max_positions_x = []

            for beam in beams_list:
                reduced_2d_grid = self._inf_matrix.get_bev_2d_grid(beam_id=beam['beam_id'])
                reduced_2d_grid = reduced_2d_grid[~np.all(reduced_2d_grid == -1, axis=1), :]  # Remove rows not in BEV

                beam['reduced_2d_grid'] = reduced_2d_grid
                beam['num_rows'] = reduced_2d_grid.shape[0]
                beam['num_cols'] = reduced_2d_grid.shape[1]
                beam['start_leaf_pair'] = np.max(beam['MLC_leaf_idx'][0])
                beam['end_leaf_pair'] = np.min(beam['MLC_leaf_idx'][0])
                beam['start_beamlet_idx'] = np.min(reduced_2d_grid[reduced_2d_grid >= 0])
                beam['end_beamlet_idx'] = np.max(reduced_2d_grid)
                beam['min_position_x_mm'] = np.min(beam['position_x_mm'][0])
                beam['max_position_x_mm'] = np.max(beam['position_x_mm'][0])

                # Store values for arc-level aggregation
                start_beamlet_idxs.append(beam['start_beamlet_idx'])
                end_beamlet_idxs.append(beam['end_beamlet_idx'])
                num_rows_list.append(beam['num_rows'])
                num_cols_list.append(beam['num_cols'])
                start_leaf_pairs.append(beam['start_leaf_pair'])
                end_leaf_pairs.append(beam['end_leaf_pair'])
                min_positions_x.append(beam['min_position_x_mm'])
                max_positions_x.append(beam['max_position_x_mm'])

            # Compute arc-level values in a single step
            arc['num_beams'] = len(beams_list)
            arc['start_beamlet_idx'] = start_beamlet_idxs[0]  # First beam
            arc['end_beamlet_idx'] = end_beamlet_idxs[-1]  # Last beam
            arc['total_rows'] = np.sum(num_rows_list)
            arc['max_rows'] = np.max(num_rows_list)
            arc['max_cols'] = np.max(num_cols_list)
            arc['start_leaf_pair'] = np.max(start_leaf_pairs)
            arc['end_leaf_pair'] = np.min(end_leaf_pairs)
            arc['min_position_x_mm'] = np.min(min_positions_x)
            arc['max_position_x_mm'] = np.max(max_positions_x)

        # align beams if beams are off in x direction
        for i, arc in enumerate((arcs_dict['arcs'])):
            for j, beam in enumerate(arc['vmat_opt']):
                beam['offset_x'] = (beam['min_position_x_mm'] - arc['min_position_x_mm']) / self._inf_matrix.beamlet_width_mm

    def get_initial_leaf_pos(self, initial_leaf_pos='BEV'):

        """
            Initialize leaf positions for the scp.
            :param initial_leaf_pos: initial leaf position. Default is BEV
            :return: None
        """
        arcs_dict = self.arcs_dict
        np.random.seed(0)
        for i, arc in enumerate((arcs_dict['arcs'])):
            beams_list = arc['vmat_opt']
            for j, beam in enumerate(beams_list):

                beam['leaf_pos_bev'] = []
                beam['leaf_pos_left'] = []
                beam['leaf_pos_right'] = []
                beam['leaf_pos_f'] = []
                beam['leaf_pos_b'] = []
                for _, row in enumerate(beam['reduced_2d_grid']):
                    if len(row) > 0:
                        left_pos_bev, right_pos_bev = np.argwhere(row >= 0)[0][0] - 1, np.argwhere(row >= 0)[-1][0] + 1

                        beam['leaf_pos_bev'].append([left_pos_bev, right_pos_bev])
                        if initial_leaf_pos.upper() == 'BEV':
                            left_pos, right_pos = left_pos_bev, right_pos_bev
                        elif initial_leaf_pos.upper() == 'RANDOM':
                            # get random leaf positions between left and right leaf
                            # set seed
                            # np.random.seed(0)
                            left_pos = np.random.randint(left_pos_bev, right_pos_bev-1)  # matlab strategy for choosing random leaf position. It is half open[low, high)
                            # np.random.seed(0)
                            right_pos = np.random.randint(left_pos + 2, right_pos_bev+1)
                        else:
                            raise ValueError("Invalid initial leaf position. Choose between BEV or random")
                        beam['leaf_pos_left'].append(left_pos)
                        beam['leaf_pos_right'].append(right_pos)
                        beam['leaf_pos_f'].append([left_pos, right_pos])
                        beam['leaf_pos_b'].append([left_pos, right_pos])


    def gen_interior_and_boundary_beamlets(self, forward_backward: int = 1, step_size_f: int = 8, step_size_b: int = 8):
        """
        Create interior and boundary beamlets based upon step_size and forward backward

        :param forward_backward: forward backward value. Default is 1. If 1, forward, if 0, backward
        :param step_size_f: step size for forward. Default is 8
        :param step_size_b: step size for backward. Default is 8
        :return: None

        """
        arcs_dict = self.arcs_dict
        for a, arc in enumerate(arcs_dict['arcs']):
            vmat = arc['vmat_opt']
            num_beams = arc['num_beams']

            for b in range(num_beams):
                bound_ind_l = []
                bound_ind_r = []
                int_ind = []

                num_rows = vmat[b]['num_rows']
                num_cols = vmat[b]['num_cols']
                map_ = vmat[b]['reduced_2d_grid']
                bev = vmat[b]['leaf_pos_bev']
                leaf_pos_l = vmat[b]['leaf_pos_left']
                leaf_pos_r = vmat[b]['leaf_pos_right']

                for r in range(num_rows):
                    # moving the left/right leaves forward
                    row = map_[r, :]
                    new_leaf_pos_l = min(leaf_pos_l[r] + step_size_f * forward_backward, leaf_pos_r[r] - 1)
                    new_leaf_pos_r = max(leaf_pos_r[r] - step_size_f * forward_backward, leaf_pos_l[r] + 1)

                    # collision check
                    count = 0
                    while new_leaf_pos_l >= new_leaf_pos_r:
                        if count % 2 == 0:
                            new_leaf_pos_l -= 1
                        else:
                            new_leaf_pos_r += 1
                        count += 1

                    # create boundary indices
                    if leaf_pos_l[r] + 1 <= new_leaf_pos_l:
                        bound_ind_l.append(list(map_[r, leaf_pos_l[r] + 1:new_leaf_pos_l + 1]))
                        # bound_ind_l.append(list(map_[r, leaf_pos_l[r]:new_leaf_pos_l]))
                    else:
                        bound_ind_l.append([])

                    if new_leaf_pos_r <= leaf_pos_r[r] - 1:
                        bound_ind_r.append(list(map_[r, new_leaf_pos_r:leaf_pos_r[r]]))
                    else:
                        bound_ind_r.append([])

                    new_leaf_pos_l = max(leaf_pos_l[r] - step_size_b * (1 - forward_backward), -1)
                    # new_leaf_pos_l = max(leaf_pos_l[r] - step_size_b * (1 - forward_backward), 0)
                    new_leaf_pos_r = min(leaf_pos_r[r] + step_size_b * (1 - forward_backward), num_cols)

                    # beam eye view check
                    new_leaf_pos_l = max(new_leaf_pos_l, bev[r][0])
                    new_leaf_pos_r = min(new_leaf_pos_r, bev[r][1])

                    if new_leaf_pos_l + 1 <= leaf_pos_l[r]:
                        if not bound_ind_l[r]:
                            bound_ind_l[r] = list(map_[r, new_leaf_pos_l + 1:leaf_pos_l[r] + 1])
                        else:
                            bound_ind_l[r].extend(map_[r, new_leaf_pos_l + 1:leaf_pos_l[r] + 1])

                    if leaf_pos_r[r] <= new_leaf_pos_r - 1:
                        if not bound_ind_r[r]:
                            bound_ind_r[r] = list(map_[r, leaf_pos_r[r]:new_leaf_pos_r])
                        else:
                            bound_ind_r[r].extend(map_[r, leaf_pos_r[r]:new_leaf_pos_r])

                    # ind = ~np.isin(map_[r, :][map_[r, :] >= 0], np.union1d(bound_ind_l[r], bound_ind_r[r]))
                    # if any(ind):
                    #     int_ind.extend(map_[r, :][map_[r, :] >= 0][ind])
                    # using matlab version
                    if bound_ind_l[r] and bound_ind_r[r]:  # Both are non-empty
                        if not int_ind:
                            int_ind = list(range(max(bound_ind_l[r]) + 1, min(bound_ind_r[r])))
                        else:
                            int_ind.extend(range(max(bound_ind_l[r]) + 1, min(bound_ind_r[r])))

                    elif not bound_ind_l[r] and not bound_ind_r[r]:  # Both are empty
                        if not int_ind:
                            if leaf_pos_l[r] + 1 == leaf_pos_r[r]:
                                int_ind = []
                            else:
                                int_ind = list(map_[r, leaf_pos_l[r] + 1:leaf_pos_r[r]])
                        else:
                            if leaf_pos_l[r] + 1 == leaf_pos_r[r]:
                                continue
                            else:
                                int_ind.extend(map_[r, leaf_pos_l[r] + 1:leaf_pos_r[r]])

                    elif not bound_ind_l[r] and bound_ind_r[r]:  # Only left boundary is empty
                        if not int_ind:
                            int_ind = list(range(map_[r, leaf_pos_l[r] + 1], min(bound_ind_r[r])))
                        else:
                            int_ind.extend(range(map_[r, leaf_pos_l[r] + 1], min(bound_ind_r[r])))

                    elif bound_ind_l[r] and not bound_ind_r[r]:  # Only right boundary is empty
                        if not int_ind:
                            int_ind = list(range(max(bound_ind_l[r]) + 1, map_[r, leaf_pos_r[r] - 1] + 1))
                        else:
                            int_ind.extend(range(max(bound_ind_l[r]) + 1, map_[r, leaf_pos_r[r] - 1] + 1))
                vmat[b]['bound_ind_left'] = bound_ind_l
                vmat[b]['bound_ind_right'] = bound_ind_r
                vmat[b]['int_ind'] = int_ind

    def calc_actual_from_intermediate_sol(self, sol: dict):
        """
        Create actual solution from intermediate solution.
        :param sol: solution dictionary
        :return: None

        """
        int_v = sol['int_v']
        bound_v_l = sol['bound_v_l']
        bound_v_r = sol['bound_v_r']
        arcs = self.arcs_dict['arcs']

        beam_so_far = 0
        beamlet_so_far = 0
        count = 0
        w_beamlet = []
        # calculate intermediate solution for interior and boundary beamlets
        for a, arc in enumerate(arcs):
            num_beams = arc['num_beams']
            num_beamlets = arc['end_beamlet_idx'] - arc['start_beamlet_idx'] + 1
            w_beamlet.append(np.zeros(num_beamlets))

            for b, beam in enumerate(arc['vmat_opt']):
                beam['bound_v_l'] = []
                beam['bound_v_r'] = []
                beam['int_v'] = int_v[beam_so_far + b]

                if beam['int_ind']:
                    w_beamlet[a][np.array(beam['int_ind']) - beamlet_so_far] = beam['int_v']

                for r in range(beam['num_rows']):
                    beam['bound_v_l'].append(bound_v_l[count])
                    if beam['bound_ind_left'][r]:
                        w_beamlet[a][np.array(beam['bound_ind_left'][r]) - beamlet_so_far] = beam['bound_v_l'][r]

                    beam['bound_v_r'].append(bound_v_r[count])
                    if beam['bound_ind_right'][r]:
                        w_beamlet[a][np.array(beam['bound_ind_right'][r]) - beamlet_so_far] = beam['bound_v_r'][r]
                    count += 1

            beam_so_far += num_beams
            beamlet_so_far += num_beamlets
            arcs[a]['w_beamlet'] = w_beamlet[a]
        self.calculate_beamlet_value()
        self.intermediate_to_actual()
        # self.update_beamlets_weights()  # first and 2nd beam weights adjustment
        self._get_leaf_pos_in_beamlet(sol=sol)

    def update_leaf_pos(self, forward_backward: int, update_reference_leaf_pos: bool = True):
        if update_reference_leaf_pos:
            self._update_reference_leaf_pos()

        for arc in self.arcs_dict['arcs']:

            for beam in arc['vmat_opt']:
                for r in range(beam['num_rows']):
                    beam['leaf_pos_left'][r] = beam['leaf_pos_f'][r][0] * forward_backward + beam['leaf_pos_b'][r][
                        0] * (1 - forward_backward)
                    beam['leaf_pos_right'][r] = beam['leaf_pos_f'][r][1] * forward_backward + beam['leaf_pos_b'][r][
                        1] * (1 - forward_backward)

    def update_best_solution(self):
        """
        Update best solution if the current solution is better than the best solution.
        :return: None
        """
        arcs = self.arcs_dict['arcs']
        for a, arc in enumerate(arcs):
            for b, beam in enumerate(arc['vmat_opt']):
                beam['best_beam_weight'] = beam['int_v']
                beam['best_leaf_position_in_cm'] = beam['cont_leaf_pos_in_beamlet'] * self._inf_matrix.beamlet_width_mm / 10
            arc['best_w_beamlet_act'] = arc['w_beamlet_act']

    def calculate_beamlet_value(self):
        """
        Calculate beamlet values between (0-1) for the intermediate solution.
        :return: None

        """
        # calculates the beamlet values between (0-1)
        arcs = self.arcs_dict['arcs']
        num_beamlets_so_far = 0

        for a, arc in enumerate(arcs):
            w_beamlet = arc['w_beamlet']
            num_beamlets = arc['end_beamlet_idx'] - arc['start_beamlet_idx'] + 1

            for b, beam in enumerate(arc['vmat_opt']):
                beam['intermediate_sol'] = np.zeros_like(beam['reduced_2d_grid'], dtype=float)

                for i in range(beam['start_beamlet_idx'], beam['end_beamlet_idx'] + 1):
                    row, col = np.where(beam['reduced_2d_grid'] == i)
                    if beam['int_v'] > 0:
                        beam['intermediate_sol'][row, col] = min(1, w_beamlet[i - num_beamlets_so_far] / beam['int_v'])
                    else:
                        beam['intermediate_sol'][row, col] = 0

            num_beamlets_so_far += num_beamlets

        return arcs

    def calculate_dose(self, inf_matrix: InfluenceMatrix, sol: dict, vmat_params: dict, best_plan: bool = False):
        """

        Calculate dose from the solution.
        :param inf_matrix: object of InfluenceMatrix class
        :param sol: solution dictionary
        :param vmat_params: vmat parameters
        :param best_plan: if True, calculate dose for best plan. Default is False
        :return: solution dictionary containing dose values

        """
        A = inf_matrix.A
        arcs = self.arcs_dict['arcs']
        adj1 = vmat_params['second_beam_adj']
        adj0 = vmat_params['first_beam_adj']
        # adj2 = vmat_params['last_beam_adj']

        if 'alpha' in vmat_params:
            alpha = vmat_params['alpha']
        else:
            alpha = 0
        if 'delta' in self._inf_matrix.opt_voxels_dict:
            delta = self._inf_matrix.opt_voxels_dict['delta'][0]
        else:
            delta = np.zeros(A.shape[0])
        if best_plan:
            sol['best_act_dose_v'] = np.zeros(A.shape[0])
        else:
            sol['act_dose_v'] = np.zeros(A.shape[0])
            sol['int_dose_v'] = np.zeros(A.shape[0])
            sol['optimal_intensity'] = np.zeros(A.shape[1])

        beamlet_so_far = 0
        sum_beamlets_act = 0
        sum_beamlets_best_act = 0
        sum_beamlets_int = 0
        for arc in arcs:
            from_ = arc['start_beamlet_idx']
            to_ = arc['end_beamlet_idx']

            num_beamlets = to_ - from_ + 1
            adjust_beamlets_weight = np.ones(num_beamlets)

            # adjust all beams weight based on gantry angle
            for b, beam in enumerate(arc['vmat_opt']):
                beam_start_idx = beam['start_beamlet_idx']
                beam_end_idx = beam['end_beamlet_idx']
                adjust_beamlets_weight[beam_start_idx - beamlet_so_far: beam_end_idx - beamlet_so_far + 1] = arc['map_adj_int'][b]

            if best_plan:
                sol['best_act_dose_v'] += A[:, from_:to_ + 1] @ (arc['best_w_beamlet_act'] * adjust_beamlets_weight)
            else:
                sol['act_dose_v'] += A[:, from_:to_ + 1] @ (arc['w_beamlet_act'] * adjust_beamlets_weight)
                # sum_beamlets_act += np.sum(arc['w_beamlet_act'] * adjust_beamlets_weight)

                sol['int_dose_v'] += A[:, from_:to_ + 1] @ (arc['w_beamlet'] * adjust_beamlets_weight)
                # sum_beamlets_int += np.sum(arc['w_beamlet'] * adjust_beamlets_weight)

                sol['optimal_intensity'][from_:to_ + 1] = arc['w_beamlet_act'] * adjust_beamlets_weight
            sum_beamlets_act += np.sum(arc['w_beamlet_act'])
            sum_beamlets_int += np.sum(arc['w_beamlet'])
            if best_plan:
                sum_beamlets_best_act += np.sum(arc['best_w_beamlet_act'])
            beamlet_so_far = beamlet_so_far + num_beamlets
        if alpha:
            if best_plan:
                sol['best_act_dose_v'] += (sum_beamlets_best_act / A.shape[1]) * alpha * delta
            else:
                sol['act_dose_v'] += (sum_beamlets_act / A.shape[1]) * alpha * delta
                sol['int_dose_v'] += (sum_beamlets_int / A.shape[1]) * alpha * delta
        sol['beamlet_mean_act'] = (sum_beamlets_act / A.shape[1])  # save for future use
        return sol

    def intermediate_to_actual(self):
        """
        Convert intermediate solution to actual feasible solution.

        """
        arcs = self.arcs_dict['arcs']
        beamlet_so_far = 0
        # Convert intermediate solution to actual feasible solution
        for a, arc in enumerate(arcs):
            num_beams = arc['num_beams']
            num_beamlets = arc['end_beamlet_idx'] - arc['start_beamlet_idx'] + 1
            w_beamlet_act = np.zeros(num_beamlets)

            for b, beam in enumerate(arc['vmat_opt']):
                num_rows = beam['num_rows']
                num_cols = beam['num_cols']
                reduced_2d_grid = beam['reduced_2d_grid']
                int_sol = beam['intermediate_sol']
                act_solution = np.zeros((num_rows, num_cols))

                for r in range(num_rows):
                    row = int_sol[r, :]
                    fractional_indices = (row > 0.0) & (row < 1.0)
                    signal = True
                    if np.sum(fractional_indices) <= 1:
                        act_solution[r, :] = row
                        signal = False
                    elif np.sum(fractional_indices) == 2:
                        col = np.where(fractional_indices)[0]
                        if col[1] - col[0] > 1:
                            act_solution[r, :] = row
                            signal = False

                    if signal:
                        act_solution[r, :] = row
                        if beam['bound_ind_left'][r]:
                            bound_ind = beam['bound_ind_left'][r]
                            col = np.where(np.isin(reduced_2d_grid[r, :], bound_ind))[0]
                            sum_boundary = np.sum(row[col])
                            count = 0
                            while np.floor(sum_boundary) >= 1:
                                c = col[-1] - count
                                act_solution[r, c] = 1
                                sum_boundary -= 1
                                count += 1
                            if sum_boundary > 0:
                                act_solution[r, col[-1] - count] = sum_boundary
                            if col[0] <= col[-1] - count - 1:
                                act_solution[r, col[0]: col[-1] - count] = 0
                        if beam['bound_ind_right'][r]:
                            bound_ind = beam['bound_ind_right'][r]
                            col = np.where(np.isin(reduced_2d_grid[r, :], bound_ind))[0]
                            sum_boundary = np.sum(row[col])
                            count = 0
                            while np.floor(sum_boundary) >= 1:
                                c = col[0] + count
                                act_solution[r, c] = 1
                                sum_boundary -= 1
                                count += 1
                            if sum_boundary > 0:
                                act_solution[r, col[0] + count] = sum_boundary
                            if col[0] + count + 1 <= col[-1]:
                                act_solution[r, col[0] + count + 1: col[-1] + 1] = 0
                    for c in range(num_cols):
                        if reduced_2d_grid[r, c] >= 0:
                            w_beamlet_act[reduced_2d_grid[r, c] - beamlet_so_far] = act_solution[r, c] * beam['int_v']

                beam['actual_sol'] = act_solution

            arc['w_beamlet_act'] = w_beamlet_act
            beamlet_so_far = beamlet_so_far + num_beamlets

    def _update_reference_leaf_pos(self):
        """
        Update reference leaf position in forward and backward direction. The following leaf positions are updated only if a solution is accepted
        they are necessary because if a solution is rejected and forward/backward
        is changed then you need to go back to this reference leaf positions to
        update your leaf positions

        """
        arcs = self.arcs_dict['arcs']
        for arc in arcs:

            for b, beam in enumerate(arc['vmat_opt']):
                int_sol = beam['intermediate_sol']
                reduced_2d_grid = beam['reduced_2d_grid']

                for r in range(beam['num_rows']):
                    if beam['bound_ind_left'][r]:
                        bound_ind = beam['bound_ind_left'][r]
                        col = np.where(np.isin(reduced_2d_grid[r, :], bound_ind))[0]
                        beam['leaf_pos_b'][r][0] = max(col) - int(sum(int_sol[r, col]))
                        beam['leaf_pos_f'][r][0] = max(col) - int(np.ceil(sum(int_sol[r, col])))
                    else:
                        beam['leaf_pos_b'][r][0] = beam['leaf_pos_left'][r]
                        beam['leaf_pos_f'][r][0] = beam['leaf_pos_left'][r]

                    if beam['bound_ind_right'][r]:
                        bound_ind = beam['bound_ind_right'][r]
                        col = np.where(np.isin(reduced_2d_grid[r, :], bound_ind))[0]
                        beam['leaf_pos_b'][r][1] = min(col) + int(sum(int_sol[r, col]))
                        beam['leaf_pos_f'][r][1] = min(col) + int(np.ceil(sum(int_sol[r, col])))
                    else:
                        beam['leaf_pos_b'][r][1] = beam['leaf_pos_right'][r]
                        beam['leaf_pos_f'][r][1] = beam['leaf_pos_right'][r]

    def _get_leaf_pos_in_beamlet(self, sol):
        """
        Get leaf position relative to beamlets.
        :param sol: solution dictionary
        :return: None

        """
        arcs = self.arcs_dict['arcs']
        leaf_pos_mu_l = sol['leaf_pos_mu_l']
        leaf_pos_mu_r = sol['leaf_pos_mu_r']
        count = 0
        beam_so_far = 0
        for a, arc in enumerate(arcs):
            for b, beam in enumerate(arc['vmat_opt']):
                num_rows = beam['num_rows']
                beam_mu = sol['int_v'][beam_so_far + b]
                beam['cont_leaf_pos_in_beamlet'] = np.zeros((num_rows, 2))
                for r in range(num_rows):
                    beam['cont_leaf_pos_in_beamlet'][r, 0] = np.round(leaf_pos_mu_l[count] / (beam_mu + 0.000000000001), 4)
                    beam['cont_leaf_pos_in_beamlet'][r, 1] = np.round(leaf_pos_mu_r[count] / (beam_mu + 0.000000000001), 4)
                    count = count + 1
            beam_so_far += arc['num_beams']
