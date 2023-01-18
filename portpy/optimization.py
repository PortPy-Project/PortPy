from __future__ import annotations
import numpy as np
import cvxpy as cp
import time
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from portpy.plan import Plan
    from portpy.influence_matrix import InfluenceMatrix


class Optimization(object):
    """
    Optimization class for optimizing and creating the plan
    """
    @staticmethod
    def run_IMRT_fluence_map_CVXPy(my_plan: Plan, inf_matrix: InfluenceMatrix = None, solver: str = 'MOSEK') -> dict:
        """
        It runs optimization to create optimal plan based upon clinical criteria

        :param my_plan: object of class Plan
        :param inf_matrix: object of class InfluenceMatrix
        :param solver: default solver 'MOSEK'. check cvxpy website for available solvers
        :return: returns the solution dictionary in format of:
            dict: {
                   'optimal_intensity': list(float),
                   'dose_1d': list(float),
                   'inf_matrix': Pointer to object of InfluenceMatrix }
                  }
        :Example:
        >>> Optimization.run_IMRT_fluence_map_CVXPy(my_plan=my_plan, inf_matrix=inf_matrix, solver='MOSEK')
        """

        t = time.time()

        # get data for optimization
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        A = inf_matrix.A
        cc_dict = my_plan.clinical_criteria.clinical_criteria_dict
        criteria = cc_dict['criteria']
        pres = cc_dict['pres_per_fraction_gy']
        num_fractions = cc_dict['num_of_fractions']
        [Qx, Qy] = Optimization.get_smoothness_matrix(inf_matrix.beamlets_dict)
        st = inf_matrix

        # create and add rind constraints
        rinds = ['RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
        if rinds[0] not in my_plan.structures.structures_dict['name']:  # check if rind is already created. If yes, skip rind creation
            Optimization.create_rinds(my_plan, size_mm=[5, 5, 20, 30, 500])
            Optimization.set_rinds_opt_voxel_idx(my_plan, inf_matrix=inf_matrix)  # rind_0 is 5mm after PTV, rind_2 is 5 mm after rind_1, and so on..
        else:
            Optimization.set_rinds_opt_voxel_idx(my_plan, inf_matrix=inf_matrix)

        rind_max_dose_perc = [1.1, 1.05, 0.9, 0.85, 0.75]
        for i, rind in enumerate(rinds):  # Add rind constraints
            parameters = {'structure_name': rind}
            total_pres = cc_dict['pres_per_fraction_gy'] * cc_dict['num_of_fractions']
            constraints = {'limit_dose_gy': total_pres * rind_max_dose_perc[i]}
            my_plan.clinical_criteria.add_criterion(criterion='max_dose', parameters=parameters,
                                                    constraints=constraints)

        # # setting weights for oar objectives
        all_vox = np.arange(A.shape[0])
        oar_voxels = all_vox[~np.isin(np.arange(A.shape[0]), st.get_opt_voxels_idx('PTV'))]
        oar_weights = np.ones(A[oar_voxels, :].shape[0])
        if cc_dict['disease_site'] == 'Prostate':
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RECT_WALL')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('BLAD_WALL')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3
        elif cc_dict['disease_site'] == 'Lung':
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('CORD')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('ESOPHAGUS')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('HEART')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3

        # Construct the problem.
        x = cp.Variable(A.shape[1], pos=True)
        dO = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)

        # Form objective.
        ptv_overdose_weight = 10000
        ptv_underdose_weight = 10  # number of times of the ptv overdose weight
        smoothness_weight = 100
        smoothness_X_weight = 0.6
        smoothness_Y_weight = 0.4

        print('Objective Start')
        obj = [ptv_overdose_weight * (1 / len(st.get_opt_voxels_idx('PTV'))) * (cp.sum_squares(dO) + ptv_underdose_weight * cp.sum_squares(dU)),
               smoothness_weight * (smoothness_X_weight * cp.sum_squares(Qx @ x) + smoothness_Y_weight * cp.sum_squares(Qy @ x))]
        # 10 * (1/infMatrix[oar_voxels, :].shape[0])*cp.sum_squares(cp.multiply(cp.sqrt(oar_weights), infMatrix[oar_voxels, :] @ w))]

        print('Objective done')
        print('Constraints Start')
        constraints = []
        # constraints += [wMean == cp.sum(w)/w.shape[0]]
        for i in range(len(criteria)):
            if 'max_dose' in criteria[i]['name']:
                if 'limit_dose_gy' in criteria[i]['constraints']:
                    limit = criteria[i]['constraints']['limit_dose_gy']
                    org = criteria[i]['parameters']['structure_name']
                    if org != 'GTV' or org != 'CTV':
                        constraints += [A[st.get_opt_voxels_idx(org), :] @ x <= limit / num_fractions]
            elif 'mean_dose' in criteria[i]['name']:
                if 'limit_dose_gy' in criteria[i]['constraints']:
                    limit = criteria[i]['constraints']['limit_dose_gy']
                    org = criteria[i]['parameters']['structure_name']
                    # mean constraints using voxel weights
                    constraints += [(1 / sum(st.get_opt_voxels_size(org))) *
                                    (cp.sum((cp.multiply(st.get_opt_voxels_size(org), A[st.get_opt_voxels_idx(org),
                                                                                     :] @ x)))) <= limit / num_fractions]

        # Step 1 and 2 constraint
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x <= pres + dO]
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x >= pres - dU]

        # Smoothness Constraint
        # for b in range(len(my_plan.beams_dict.beams_dict['ID'])):
        #     startB = my_plan.beams_dict.beams_dict['start_beamlet'][b]
        #     endB = my_plan.beams_dict.beams_dict['end_beamlet'][b]
        #     constraints += [0.6 * cp.sum_squares(
        #         Qx[startB:endB, startB:endB] @ x[startB:endB]) + 0.4 * cp.sum_squares(
        #         Qy[startB:endB, startB:endB] @ x[startB:endB]) <= 0.5]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)

        print('Problem loaded')
        prob.solve(solver=solver, verbose=True)
        print("optimal value with MOSEK:", prob.value)
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))

        # saving optimal solution to the solution dictionary
        sol = {'optimal_intensity': x.value.astype('float32'), 'inf_matrix': inf_matrix}

        return sol

    @staticmethod
    def run_IMRT_fluence_map_CVXPy_dvh_benchmark(my_plan, inf_matrix=None, solver='MOSEK'):
        """
        Add dvh constraints and solve the optimization for getting ground truth solution

        :param inf_matrix: object of class InfluenceMatrix
        :param my_plan: object of class Plan
        :param solver: default solver 'MOSEK'. check cvxpy website for available solvers
        :return: save optimal solution to plan object called opt_sol


        """
        t = time.time()

        # get data for optimization
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        A = inf_matrix.A
        cc_dict = my_plan.clinical_criteria.clinical_criteria_dict
        criteria = cc_dict['criteria']
        pres = cc_dict['pres_per_fraction_gy']
        num_fractions = cc_dict['num_of_fractions']
        [Qx, Qy] = Optimization.get_smoothness_matrix(inf_matrix.beamlets_dict)
        st = inf_matrix

        # create and add rind constraints
        rinds = ['RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
        if rinds[0] not in my_plan.structures.structures_dict['name']:  # check if rind is already created. If yes, skip rind creation
            Optimization.create_rinds(my_plan, size_mm=[5, 5, 20, 30, 500])
            Optimization.set_rinds_opt_voxel_idx(my_plan,
                                                 inf_matrix=inf_matrix)  # rind_0 is 5mm after PTV, rind_2 is 5 mm after rind_1, and so on..
        else:
            Optimization.set_rinds_opt_voxel_idx(my_plan, inf_matrix=inf_matrix)

        rind_max_dose_perc = [1.1, 1.0, 0.9, 0.7, 0.65]  # add rind constraints to the problem
        for i, rind in enumerate(rinds):
            parameters = {'structure_name': rind}
            total_pres = cc_dict['pres_per_fraction_gy'] * cc_dict['num_of_fractions']
            constraints = {'limit_dose_gy': total_pres * rind_max_dose_perc[i]}
            my_plan.clinical_criteria.add_criterion(criterion='max_dose', parameters=parameters,
                                                    constraints=constraints)

        # setting weights for oar objectives
        all_vox = np.arange(A.shape[0])
        oar_voxels = all_vox[~np.isin(np.arange(A.shape[0]), st.get_opt_voxels_idx('PTV'))]
        oar_weights = np.ones(A[oar_voxels, :].shape[0])
        if cc_dict['disease_site'] == 'Prostate':
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RECT_WALL')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('BLAD_WALL')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3
        elif cc_dict['disease_site'] == 'Lung':
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('CORD')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('ESOPHAGUS')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('HEART')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3

        # get volume constraint in V(Gy) <= v% format
        import pandas as pd
        df = pd.DataFrame()
        count = 0
        for i in range(len(criteria)):
            if 'dose_volume' in criteria[i]['name']:
                limit_key = Optimization.matching_keys(criteria[i]['constraints'], 'limit')
                if limit_key in criteria[i]['constraints']:
                    df.at[count, 'structure'] = criteria[i]['parameters']['structure_name']
                    df.at[count, 'dose_gy'] = criteria[i]['parameters']['dose_gy']

                    # getting max dose_1d for the same structure
                    max_dose_struct = 1000
                    for j in range(len(criteria)):
                        if 'max_dose' in criteria[j]['name']:
                            if 'limit_dose_gy' in criteria[j]['constraints']:
                                org = criteria[j]['parameters']['structure_name']
                                if org == criteria[i]['parameters']['structure_name']:
                                    max_dose_struct = criteria[j]['constraints']['limit_dose_gy']
                    df.at[count, 'M'] = max_dose_struct - criteria[i]['parameters']['dose_gy']
                    if 'perc' in limit_key:
                        df.at[count, 'vol_perc'] = criteria[i]['constraints'][limit_key]
                    count = count + 1
        # Construct the problem.
        x = cp.Variable(A.shape[1], pos=True)
        dO = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)

        # binary variable for dvh constraint
        b = cp.Variable(len(np.concatenate([st.get_opt_voxels_idx(org) for org in df.structure.to_list()])),
                        boolean=True)
        # Form objective.
        ptv_overdose_weight = 10000
        ptv_underdose_weight = 10  # number of times of the ptv overdose weight
        smoothness_weight = 1000
        smoothness_X_weight = 0.6
        smoothness_Y_weight = 0.4

        print('Objective Start')
        obj = [ptv_overdose_weight * (1 / len(st.get_opt_voxels_idx('PTV'))) * (
                    cp.sum_squares(dO) + ptv_underdose_weight * cp.sum_squares(dU)),
               smoothness_weight * (
                           smoothness_X_weight * cp.sum_squares(Qx @ x) + smoothness_Y_weight * cp.sum_squares(Qy @ x))]
        # 10 * (1/infMatrix[oar_voxels, :].shape[0])*cp.sum_squares(cp.multiply(cp.sqrt(oar_weights), infMatrix[oar_voxels, :] @ w))]

        print('Objective done')
        print('Constraints Start')
        constraints = []
        # constraints += [wMean == cp.sum(w)/w.shape[0]]
        for i in range(len(criteria)):
            if 'max_dose' in criteria[i]['name']:
                if 'limit_dose_gy' in criteria[i]['constraints']:
                    limit = criteria[i]['constraints']['limit_dose_gy']
                    org = criteria[i]['parameters']['structure_name']
                    if org != 'GTV' or org != 'CTV':
                        constraints += [A[st.get_opt_voxels_idx(org), :] @ x <= limit / num_fractions]
            elif 'mean_dose' in criteria[i]['name']:
                if 'limit_dose_gy' in criteria[i]['constraints']:
                    limit = criteria[i]['constraints']['limit_dose_gy']
                    org = criteria[i]['parameters']['structure_name']
                    # constraints += [
                    #     (1 / len(st.get_voxels_idx(org))) * (cp.sum(infMatrix[st.get_voxels_idx(org), :] @ w)) <=
                    #     limit / num_fractions]

                    # mean constraints using voxel weights
                    constraints += [(1 / sum(st.get_opt_voxels_size(org))) *
                                    (cp.sum((cp.multiply(st.get_opt_voxels_size(org), A[st.get_opt_voxels_idx(org),
                                                                                      :] @ x)))) <= limit / num_fractions]
        start = 0
        for i in range(len(df)):
            struct, limit, v, M = df.loc[i, 'structure'], df.loc[i, 'dose_gy'], df.loc[i, 'vol_perc'], df.loc[i, 'M']
            end = start + len(st.get_opt_voxels_idx(struct))
            constraints += [A[st.get_opt_voxels_idx(struct), :] @ x <= limit / num_fractions + b[start:end] * M]
            constraints += [cp.sum(cp.multiply(b[start:end], st.get_opt_voxels_size(struct))) <= v / 100 * cp.sum(
                st.get_opt_voxels_size(struct))]
            start = end

        # Step 1 and 2 constraint
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x <= pres + dO]
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x >= pres - dU]

        # Smoothness Constraint
        # for b in range(len(my_plan.beams_dict.beams_dict['ID'])):
        #     startB = my_plan.beams_dict.beams_dict['start_beamlet'][b]
        #     endB = my_plan.beams_dict.beams_dict['end_beamlet'][b]
        #     constraints += [0.6 * cp.sum_squares(
        #         Qx[startB:endB, startB:endB] @ x[startB:endB]) + 0.4 * cp.sum_squares(
        #         Qy[startB:endB, startB:endB] @ x[startB:endB]) <= 0.5]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)
        # Defining the constraints
        print('Problem loaded')
        prob.solve(solver=solver, verbose=True)
        print("optimal value with MOSEK:", prob.value)
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))

        # saving optimal solution to my_plan
        sol = {'optimal_intensity': x.value, 'dose_1d': A * x.value * num_fractions, 'inf_matrix': inf_matrix}

        return sol

    @staticmethod
    def get_smoothness_matrix(beamReq: List[dict]) -> (np.ndarray, np.ndarray):
        """
        Create smoothness matrix so that adjacent beamlets are smooth out to reduce MU

        :param beamReq: beamlets dictionary from the object of influence matrix class
        :returns: tuple(Qx, Qy) where
            Qx: first matrix have values 1 and -1 for neighbouring beamlets in X direction
            Qy: second matrix with values 1 and -1 for neighbouring beamlets in Y direction

        :Example:
        Qx = [[1 -1 0 0 0 0]
              [0 0 1 -1 0 0]
              [0 0 0 0 1 -1]]

        """
        sRow = np.zeros((beamReq[-1]['end_beamlet'] + 1, beamReq[-1]['end_beamlet'] + 1), dtype=int)
        sCol = np.zeros((beamReq[-1]['end_beamlet'] + 1, beamReq[-1]['end_beamlet'] + 1), dtype=int)
        for b in range(len(beamReq)):
            beam_map = beamReq[b]['beamlet_idx_2dgrid']

            rowsNoRepeat = [0]
            for i in range(1, np.size(beam_map, 0)):
                if (beam_map[i, :] != beam_map[rowsNoRepeat[-1], :]).any():
                    rowsNoRepeat.append(i)
            colsNoRepeat = [0]
            for j in range(1, np.size(beam_map, 1)):
                if (beam_map[:, j] != beam_map[:, colsNoRepeat[-1]]).any():
                    colsNoRepeat.append(j)
            beam_map = beam_map[np.ix_(np.asarray(rowsNoRepeat), np.asarray(colsNoRepeat))]
            for r in range(np.size(beam_map, 0)):
                startCol = 0
                endCol = np.size(beam_map, 1) - 2
                while (beam_map[r, startCol] == -1) and (startCol <= endCol):
                    startCol = startCol + 1
                while (beam_map[r, endCol] == -1) and (startCol <= endCol):
                    endCol = endCol - 1

                for c in range(startCol, endCol + 1):
                    ind = beam_map[r, c]
                    RN = beam_map[r, c + 1]
                    if ind * RN >= 0:
                        sRow[ind, ind] = int(1)
                        sRow[ind, RN] = int(-1)

            for c in range(np.size(beam_map, 1)):
                startRow = 0
                endRow = np.size(beam_map, 0) - 2
                while (beam_map[startRow, c] == -1) and (startRow <= endRow):
                    startRow = startRow + 1
                while (beam_map[endRow, c] == -1) and (startRow <= endRow):
                    endRow = endRow - 1
                for r in range(startRow, endRow + 1):
                    ind = beam_map[r, c]
                    DN = beam_map[r + 1, c]
                    if ind * DN >= 0:
                        sCol[ind, ind] = int(1)
                        sCol[ind, DN] = int(-1)
        return sRow, sCol

    @staticmethod
    def create_rinds(my_plan, inf_matrix=None, size_mm: list = None, base_structure='PTV'):
        """
        :param inf_matrix: object of class inf_matrix
        :param my_plan: object of class Plan
        :param size_mm: size in mm. default size = [5,5,20,30, inf]. It means rind_0 is 5mm after PTV, rind_2 is 5 mm after rind_1, and so on..
        :param base_structure: structure from which rinds should be created
        :return: save rinds to plan object
        """
        if size_mm is None:
            size_mm = [5, 5, 20, 30, 500]
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        print('creating rinds of size {} mm ..'.format(size_mm))
        if isinstance(size_mm, np.ndarray):
            size_mm = list(size_mm)
        ct_to_dose_map = inf_matrix.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        dose_mask = ct_to_dose_map >= 0
        dose_mask = dose_mask.astype(int)
        my_plan.structures.create_structure('dose_mask', dose_mask)

        for ind, s in enumerate(size_mm):
            rind_name = 'RIND_{}'.format(ind)
            dummy_name = 'dummy_{}'.format(ind)
            if ind == 0:
                my_plan.structures.expand(base_structure, margin_mm=s, new_structure=dummy_name)
                my_plan.structures.subtract(dummy_name, base_structure, str1_sub_str2=rind_name)
                # inf_matrix.set_opt_voxel_idx(my_plan, struct=rind_name)
            else:
                # prev_rind_name = 'RIND_{}'.format(ind - 1)
                prev_dummy_name = 'dummy_{}'.format(ind-1)
                my_plan.structures.expand(prev_dummy_name, margin_mm=s, new_structure=dummy_name)
                my_plan.structures.subtract(dummy_name, prev_dummy_name, str1_sub_str2=rind_name)
                my_plan.structures.delete_structure(prev_dummy_name)
                # for the last delete dummy
                if ind == len(size_mm) - 1:
                    my_plan.structures.delete_structure(dummy_name)  # delete dummy structure for last rind
            my_plan.structures.intersect(rind_name, 'dose_mask', str1_and_str2=rind_name)

        my_plan.structures.delete_structure('dose_mask')
                # inf_matrix.set_opt_voxel_idx(my_plan, struct=rind_name)
        print('rinds created!!')

    @staticmethod
    def set_rinds_opt_voxel_idx(plan_obj: Plan, inf_matrix: InfluenceMatrix):
        """

        :param plan_obj: object of class plan
        :param inf_matrix: object of class influence matrix
        :return: set rinds index based upon the influence matrix
        """
        rinds = [rind for idx, rind in enumerate(plan_obj.structures.structures_dict['name']) if 'RIND' in rind]
        for rind in rinds:
            inf_matrix.set_opt_voxel_idx(plan_obj, structure_name=rind)

    @staticmethod
    def matching_keys(dictionary, search_string):
        get_key = None
        for key, val in dictionary.items():
            if search_string in key:
                get_key = key
        if get_key is not None:
            return get_key
        else:
            return ''
