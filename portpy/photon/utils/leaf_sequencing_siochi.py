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

"""

This leaf sequencing algorithm is based upon intensity modulated beams with multiple static segments according to
Siochi (1999) International Journal of Radiation Oncology * Biology * Physics. This code has been inspired by
MatRad implementation by Eric Christiansen, Emily Heath, and Tong Xu.


"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from portpy.photon.plan import Plan
import numpy as np
from copy import deepcopy


def leaf_sequencing_siochi(my_plan: Plan, sol: dict, num_of_levels: int = 40) -> dict:
    """

    Create leaf sequences based upon optimal sol and returns dictionary containin leaf position and MU.

    :param my_plan: object of class Plan
    :param sol: solution dictionary
    :param num_of_levels: number of stratification levels. Default 40
    """

    beams = my_plan.beams
    inf_matrix = my_plan.inf_matrix
    leaf_sequencing = {}
    w_approx = np.zeros(inf_matrix.A.shape[1])
    for b, beam_id in enumerate(beams.get_all_beam_ids()):
        gantry_angle = beams.get_gantry_angle(beam_id=beam_id)
        leaf_sequencing[gantry_angle] = {}

        maps = inf_matrix.get_bev_2d_grid(beam_id=beam_id, finest_grid=False)
        numRows = np.size(maps, 0)
        numCols = np.size(maps, 1)
        w_maps = np.zeros((numRows, numCols))
        # wMaps = np.zeros((numRows, numCols))
        for r in range(numRows):
            for c in range(numCols):
                if maps[r, c] >= 0:
                    curr = maps[r, c]
                    w_maps[r, c] = sol['optimal_intensity'][curr]

        # create levels for fluence
        cal_fac = np.amax(w_maps)
        w_maps[w_maps == -1] = 0
        D_k = np.round(w_maps / cal_fac * num_of_levels)

        # Save

        D_0 = deepcopy(D_k)
        shapes = np.zeros((10000, D_k.shape[0], D_k.shape[1]), dtype='int')
        shapes_weight = np.zeros(10000)
        k = 0

        ind = np.where(D_k > 0)
        min_row = ind[0].min()
        max_row = ind[0].max()
        min_col = ind[1].min()
        max_col = ind[1].max()

        tops = np.zeros(D_k.shape, dtype='int')
        bases = np.zeros(D_k.shape, dtype='int')

        for i in range(min_col, max_col + 1):
            max_top = -1
            t_and_g = 1
            for j in range(min_row, max_row + 1):
                if i == min_col:
                    bases[j, i] = 1
                    tops[j, i] = bases[j, i] + D_k[j, i] - 1
                else:
                    if D_k[j, i] >= D_k[j, i - 1]:  # current rod greater than previous rod
                        bases[j, i] = bases[j, i - 1]
                        tops[j, i] = bases[j, i] + D_k[j, i] - 1
                    else:
                        if D_k[j, i] == 0:# rod length=0, put in in next slab after top of previous
                            bases[j, i] = tops[j, i - 1] + 1
                            tops[j, i] = bases[j, i] - 1
                        else:  # rod length ~ = 0, match tops
                            tops[j, i] = tops[j, i - 1]
                            bases[j, i] = tops[j, i] - D_k[j, i] + 1
                # determine which rod has the highest top in column
                if tops[j, i] > max_top:
                    max_top = tops[j, i]
                    top_row = j
            # check for collision and tongue and groove error
            while t_and_g:
                for j in range(top_row - 1, min_row-1, -1):
                    if D_k[j, i] < D_k[j + 1, i]:
                        if tops[j, i] > tops[j + 1, i]:
                            tops[j + 1, i] = tops[j, i]
                            bases[j + 1, i] = tops[j + 1, i] - D_k[j + 1, i] + 1
                        elif bases[j, i] < bases[j + 1, i]:
                            bases[j, i] = bases[j + 1, i]
                            tops[j, i] = bases[j, i] + D_k[j, i] - 1
                    else:
                        if tops[j, i] < tops[j + 1, i]:
                            tops[j, i] = tops[j + 1, i]
                            bases[j, i] = tops[j, i] - D_k[j, i] + 1
                        elif bases[j, i] > bases[j + 1, i]:
                            bases[j + 1, i] = bases[j, i]
                            tops[j + 1, i] = bases[j + 1, i] + D_k[j + 1, i] - 1

                # checking from max row to up
                for j in range(top_row + 1, max_row + 1):
                    if D_k[j, i] < D_k[j - 1, i]:
                        if tops[j, i] > tops[j - 1, i]:
                            tops[j - 1, i] = tops[j, i]
                            bases[j - 1, i] = tops[j - 1, i] - D_k[j - 1, i] + 1
                        elif bases[j, i] < bases[j - 1, i]:
                            bases[j, i] = bases[j - 1, i]
                            tops[j, i] = bases[j, i] + D_k[j, i] - 1
                    else:
                        if tops[j, i] < tops[j - 1, i]:
                            tops[j, i] = tops[j - 1, i]
                            bases[j, i] = tops[j, i] - D_k[j, i] + 1
                        elif bases[j, i] > bases[j - 1, i]:
                            bases[j - 1, i] = bases[j, i]
                            tops[j - 1, i] = bases[j - 1, i] + D_k[j - 1, i] - 1

                # check if t and g has been removed
                t_and_g = 0
                for j in range(min_row + 1, max_row + 1):
                    if D_k[j, i] < D_k[j - 1, i]:
                        if tops[j, i] > tops[j - 1, i]:
                            t_and_g = 1
                        elif bases[j, i] < bases[j-1, i]:
                            t_and_g = 1

                    else:
                        if tops[j, i] < tops[j - 1, i]:
                            t_and_g = 1
                        elif bases[j, i] > bases[j - 1, i]:
                            t_and_g = 1

        # convert to segments
        levels = np.amax(tops)
        for level in range(1, levels + 1):
            if siochi_diff_slab(tops, bases, level):
                shape_k = np.multiply(bases <= level, level <= tops)
                shapes[k, :, :] = shape_k
                k = k + 1
            shapes_weight[k-1] = shapes_weight[k-1] + 1

        leaf_shapes = shapes[0:k, :, :]
        leaf_weight = shapes_weight[0:k] / num_of_levels * cal_fac

        y_leaf_pos = beams.beams_dict['MLC_leaves_pos_y_mm'][0]
        sum_of_beam = np.zeros((numRows, numCols))
        for l in range(leaf_shapes.shape[0]):
            leaf_pos = np.empty((len(y_leaf_pos), 2), dtype='object')
            control_point = leaf_shapes[l, :, :]
            beamlet_idx_2d = control_point*(maps+1)
            beamlet_idx_2d = beamlet_idx_2d - 1  # adding and subtract 1
            for i in range(control_point.shape[0]):
                row = beamlet_idx_2d[i, :][beamlet_idx_2d[i, :] >= 0]

                if len(row) > 0:
                    first_beamlet = inf_matrix.get_beamlet_info(row[0])
                    last_beamlet = inf_matrix.get_beamlet_info(row[-1])
                    left_leaf_position = first_beamlet['position_x_mm'][0] - first_beamlet['width_mm'][0]/2
                    right_leaf_position = last_beamlet['position_x_mm'][0] + last_beamlet['width_mm'][0] / 2
                    ind = np.where(y_leaf_pos == int(first_beamlet['position_y_mm'][0]))[0][0]
                    ind_rev = len(y_leaf_pos) - ind - 1
                    leaf_pos[ind_rev, 0] = left_leaf_position
                    leaf_pos[ind_rev, 1] = right_leaf_position
            leaf_pos[:, 0][leaf_pos[:, 0] == None] = 43.5
            leaf_pos[:, 1][leaf_pos[:, 1] == None] = 44
            a = leaf_pos.flatten('F')
            string_list = [str(item) for item in a.tolist()]
            leaf_sequencing[gantry_angle].setdefault('leaf_postions', []).append(string_list)
            leaf_sequencing[gantry_angle].setdefault('MU', []).append(leaf_weight[l])
            leaf_sequencing[gantry_angle].setdefault('leaf_shapes', []).append(leaf_shapes[l])
            sum_of_beam = sum_of_beam + leaf_shapes[l]*leaf_weight[l]
        for r in range(numRows):
            for c in range(numCols):
                if maps[r, c] >= 0:
                    curr = maps[r, c]
                    w_approx[curr] = sum_of_beam[r, c]
    leaf_sequencing['optimal_intensity'] = w_approx
    return leaf_sequencing


def siochi_diff_slab(tops, bases, level):
    if level == 1:
        diff_slab = 1
    else:
        shape_level = np.multiply((bases <= level), level <= tops)
        shape_level_prev = np.multiply((bases <= level-1), (level-1 <= tops))
        diff_slab = not np.array_equiv(shape_level, shape_level_prev)
    return diff_slab

