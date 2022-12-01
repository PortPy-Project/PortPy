import numpy as np
import cvxpy as cp
import time
from portpy.beam import Beams
from portpy.structures import Structures
from portpy.clinical_criteria import ClinicalCriteria


class Optimization:
    def __init__(self, beams: Beams, structures: Structures, clinical_criteria: ClinicalCriteria):
        self.structures = structures
        self.clinical_criteria = clinical_criteria
        self.beams = beams

    def run_IMRT_optimization(self, solver='MOSEK'):
        t = time.time()

        # get data for optimization
        infMatrix = self.beams.get_influence_matrix(beam_ids=self.beams.beams_dict['ID'])
        cc_dict = self.clinical_criteria.clinical_criteria_dict
        criteria = cc_dict['criteria']
        pres = cc_dict['pres_per_fraction_gy']
        num_fractions = cc_dict['num_of_fractions']
        [X, Y] = self.get_smoothness_matrix(self.beams.beams_dict)
        st = self.structures

        # create and add rind constraints
        self.create_rinds(size_mm=[5, 5, 20, 30, 500])
        rinds = ['RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
        perc_of_pres = [1.1, 1.0, 0.9, 0.7, 0.65]
        for i, rind in enumerate(rinds):
            parameters = {'structure_name': rind}
            total_pres = cc_dict['pres_per_fraction_gy'] * cc_dict['num_of_fractions']
            constraints = {'limit_dose_gy': total_pres * perc_of_pres[i]}
            self.clinical_criteria.add_criterion(criterion='max_dose', parameters=parameters,
                                                 constraints=constraints)

        # voxel weights for oar objectives
        all_vox = np.arange(infMatrix.shape[0])
        oar_voxels = all_vox[~np.isin(np.arange(infMatrix.shape[0]), st.get_voxels_idx('PTV'))]
        oar_weights = np.ones(infMatrix[oar_voxels, :].shape[0])
        if cc_dict['disease_site'] == 'Prostate':
            oar_weights[np.where(np.isin(oar_voxels, st.get_voxels_idx('RECT_WALL')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_voxels_idx('BLAD_WALL')))] = 5
            oar_weights[np.where(np.isin(oar_voxels, st.get_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_voxels_idx('RIND_1')))] = 3
            # oar_weights[st.get_voxels_idx('PTV')] = 0
            # oar_weights[st.get_voxels_idx('RECT_WALL')] = 20
            # oar_weights[st.get_voxels_idx('BLAD_WALL')] = 5
            # oar_weights[st.get_voxels_idx('RIND_0')] = 3
            # oar_weights[st.get_voxels_idx('RIND_1')] = 3

        # Construct the problem.
        w = cp.Variable(infMatrix.shape[1], pos=True)
        dO = cp.Variable(len(st.get_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_voxels_idx('PTV')), pos=True)

        # Form objective.
        print('Objective Start')
        obj = [10000 * (1 / len(st.get_voxels_idx('PTV'))) * (cp.sum_squares(dO) + 10 * cp.sum_squares(dU)),
               1000 * (0.6 * cp.sum_squares(X @ w) + 0.4 * cp.sum_squares(Y @ w))]
               # 10 * (1/infMatrix[oar_voxels, :].shape[0])*cp.sum_squares(cp.multiply(cp.sqrt(oar_weights), infMatrix[oar_voxels, :] @ w))]

        ##Step 1 objective

        ##obj.append((1/sum(pars['points'][(getVoxels(myPlan,'PTV'))-1, 3]))*cp.sum_squares(cp.multiply(cp.sqrt(pars['points'][(getVoxels(myPlan,'PTV'))-1, 3]), infMatrix[getVoxels(myPlan,'PTV')-1, :] @ w + wMean*pars['alpha']*pars['delta'][getVoxels(myPlan,'PTV')-1] - pars['presPerFraction'])))

        ##Smoothing objective function

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
                        constraints += [infMatrix[st.get_voxels_idx(org), :] @ w <= limit / num_fractions]
            if 'mean_dose' in criteria[i]['name']:
                if 'limit_dose_gy' in criteria[i]['constraints']:
                    limit = criteria[i]['constraints']['limit_dose_gy']
                    org = criteria[i]['parameters']['structure_name']
                    # constraints += [
                    #     (1 / len(st.get_voxels_idx(org))) * (cp.sum(infMatrix[st.get_voxels_idx(org), :] @ w)) <=
                    #     limit / num_fractions]

                    # mean constraints using voxel weights
                    constraints += [(1 / sum(st.get_voxels_weights(org))) *
                                    (cp.sum((cp.multiply(st.get_voxels_weights(org), infMatrix[st.get_voxels_idx(org), :] @ w)))) <= limit / num_fractions]

        # Step 1 and 2 constraint
        constraints += [infMatrix[st.get_voxels_idx('PTV'), :] @ w <= pres + dO]
        constraints += [infMatrix[st.get_voxels_idx('PTV'), :] @ w >= pres - dU]

        # Smoothness Constraint
        for b in range(len(self.beams.beams_dict['ID'])):
            startB = self.beams.beams_dict['start_beamlet'][b]
            endB = self.beams.beams_dict['end_beamlet'][b]
            constraints += [0.6 * cp.sum_squares(
                X[startB:endB, startB:endB] @ w[startB:endB]) + 0.4 * cp.sum_squares(
                Y[startB:endB, startB:endB] @ w[startB:endB]) <= 0.5]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)
        # Defining the constraints
        print('Problem loaded')
        prob.solve(solver=solver, verbose=True)
        print("optimal value with MOSEK:", prob.value)
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))
        # self.optimal_intensity = w.value
        self.beams.optimal_intensity = w.value
        self.structures.opt_voxels_dict['dose_1d'] = infMatrix*w.value

    @staticmethod
    def get_smoothness_matrix(beamReq):
        sRow = np.zeros((beamReq['end_beamlet'][-1] + 1, beamReq['end_beamlet'][-1] + 1), dtype=int)
        sCol = np.zeros((beamReq['end_beamlet'][-1] + 1, beamReq['end_beamlet'][-1] + 1), dtype=int)
        for b in range(len(beamReq['ID'])):
            map = beamReq['beamlet_idx_2dgrid'][b]

            rowsNoRepeat = [0]
            for i in range(1, np.size(map, 0)):
                if (map[i, :] != map[rowsNoRepeat[-1], :]).any():
                    rowsNoRepeat.append(i)
            colsNoRepeat = [0]
            for j in range(1, np.size(map, 1)):
                if (map[:, j] != map[:, colsNoRepeat[-1]]).any():
                    colsNoRepeat.append(j)
            map = map[np.ix_(np.asarray(rowsNoRepeat), np.asarray(colsNoRepeat))]
            for r in range(np.size(map, 0)):
                startCol = 0
                endCol = np.size(map, 1) - 2
                while (map[r, startCol] == -1) and (startCol <= endCol):
                    startCol = startCol + 1
                while (map[r, endCol] == -1) and (startCol <= endCol):
                    endCol = endCol - 1

                for c in range(startCol, endCol + 1):
                    ind = map[r, c]
                    RN = map[r, c + 1]
                    if ind * RN >= 0:
                        sRow[ind, ind] = int(1)
                        sRow[ind, RN] = int(-1)

            for c in range(np.size(map, 1)):
                startRow = 0
                endRow = np.size(map, 0) - 2
                while (map[startRow, c] == -1) and (startRow <= endRow):
                    startRow = startRow + 1
                while (map[endRow, c] == -1) and (startRow <= endRow):
                    endRow = endRow - 1
                for r in range(startRow, endRow + 1):
                    ind = map[r, c]
                    DN = map[r + 1, c]
                    if ind * DN >= 0:
                        sCol[ind, ind] = int(1)
                        sCol[ind, DN] = int(-1)
        return sRow, sCol

    def create_rinds(self, base_structure='PTV', size_mm=None):

        if size_mm is None:
            size_mm = [5, 5, 20, 30, 500]
        print('creating rinds of size {} mm ..'.format(size_mm))
        if isinstance(size_mm, np.ndarray):
            size_mm = list(size_mm)
        for ind, s in enumerate(size_mm):
            rind_name = 'RIND_{}'.format(ind)

            if ind == 0:
                self.structures.expand(base_structure, margin_mm=s, new_structure=rind_name)
                self.structures.subtract(rind_name, base_structure, str1_sub_str2=rind_name)
                # mask_3d_modified = self.add_subtract_margin(structure=base_structure, margin_mm=5, new_structure=rind_name)
                # self.subtract(structure_1=rind_name, structure_2=base_structure)
                # mask_3d_modified = self.structure(base_structure) + s
                # self.add_structure(rind_name, mask_3d_modified)
                # mask_3d_modified = self.structure(rind_name) - self.structure(base_structure)
                # self.modify_structure(rind_name, mask_3d_modified)
            else:
                prev_rind_name = 'RIND_{}'.format(ind - 1)
                self.structures.expand(prev_rind_name, margin_mm=s, new_structure=rind_name)
                for j in range(ind):
                    rind_subtract = 'RIND_{}'.format(j)
                    if j == 0:
                        self.structures.union(rind_subtract, base_structure, str1_or_str2='dummy')
                    else:
                        self.structures.union(rind_subtract, 'dummy', str1_or_str2='dummy')
                self.structures.subtract(rind_name, 'dummy', str1_sub_str2=rind_name)
                self.structures.delete_structure('dummy')
                if ind == len(size_mm)-1:
                    ct_to_dose_map = self.structures.opt_voxels_dict['ct_to_dose_voxel_map'][0]
                    dose_mask = ct_to_dose_map >= 0
                    dose_mask = dose_mask.astype(int)
                    self.structures.create_structure('dose_mask', dose_mask)
                    self.structures.intersect(rind_name, 'dose_mask', str1_and_str2=rind_name)
                    self.structures.delete_structure('dose_mask')
        print('rinds created!!')
