import numpy as np
import scipy
from utils import load_metadata, load_data
from .beam import Beam, Beams
from .structures import Structures
import os
import pandas as pd
import cvxpy as cp
import time


class Plan(object):
    """
    A class representing plan for given plan.
    """

    # def __init__(self, ID, gantry_angle=None, collimator_angle=None, iso_center=None, beamlet_map_2d=None,
    #              BEV_structure_mask=None, beamlet_width_mm=None, beamlet_height_mm=None, beam_modality=None, MLC_type=None):
    def __init__(self, patient_name, beam_indices=None, options=None):

        # patient_name = 'ECHO_PROST_1'
        patient_folder_path = os.path.join(os.getcwd(), "..", 'Data', patient_name)
        # read all the meta data for the required patient
        meta_data = load_metadata(patient_folder_path)

        # load data for the given beam_indices
        if len(options) != 0:
            if 'loadInfluenceMatrixFull' in options and not options['loadInfluenceMatrixFull']:
                meta_data['beams']['influenceMatrixFull_File'] = [None] * len(
                    meta_data['beams']['influenceMatrixFull_File'])
            if 'loadInfluenceMatrixSparse' in options and not options['loadInfluenceMatrixSparse']:
                meta_data['beams']['influenceMatrixSparse_File'] = [None] * len(
                    meta_data['beams']['influenceMatrixSparse_File'])
            if 'loadBeamEyeViewStructureMask' in options and not options['loadBeamEyeViewStructureMask']:
                meta_data['beams']['beamEyeViewStructureMask_File'] = [None] * len(
                    meta_data['beams']['beamEyeViewStructureMask_File'])

        if beam_indices is None:
            beam_indices = [0, 10, 20, 30]
        my_plan = meta_data.copy()
        del my_plan['beams']
        beamReq = dict()
        inds = []
        for i in range(len(beam_indices)):
            if beam_indices[i] in meta_data['beams']['ID']:
                ind = np.where(np.array(meta_data['beams']['ID']) == beam_indices[i])
                ind = ind[0][0]
                inds.append(ind)
                for key in meta_data['beams']:
                    beamReq.setdefault(key, []).append(meta_data['beams'][key][ind])
        my_plan['beams'] = beamReq
        if len(inds) < len(beam_indices):
            print('some indices are not available')
        my_plan = load_data(my_plan, my_plan['patient_folder_path'])
        # df = pd.DataFrame.from_dict(my_plan['beams'])
        # self.beam = [Beam(df.loc[i]) for i in range(len(beam_indices))]
        self.beams = Beams(my_plan['beams'])
        self.structures = Structures(my_plan['structures'], my_plan['opt_voxels'])
        self.patient_name = patient_name
        self.clinical_criteria = my_plan['clinical_criteria']

    def run_optimization(self):
        t = time.time()

        infMatrix = self.beams.get_influence_matrix(beam_ids=self.beams.beams_dict['ID'])
        clinical_constraints = self.clinical_criteria['constraints']
        pres = self.clinical_criteria['pres_per_fraction_Gy']
        num_fractions = self.clinical_criteria['num_of_fractions']
        # [X, Y] = self.get_smoothness_matrix(self.beams.beams_dict['ID'])
        st = self.structures
        # Construct the problem.
        w = cp.Variable(infMatrix.shape[1], pos=True)
        dO = cp.Variable(len(st.get_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_voxels_idx('PTV')), pos=True)
        # Form objective.
        print('Objective Start')
        obj = []

        ##Step 1 objective

        ##obj.append((1/sum(pars['points'][(getVoxels(myPlan,'PTV'))-1, 3]))*cp.sum_squares(cp.multiply(cp.sqrt(pars['points'][(getVoxels(myPlan,'PTV'))-1, 3]), infMatrix[getVoxels(myPlan,'PTV')-1, :] @ w + wMean*pars['alpha']*pars['delta'][getVoxels(myPlan,'PTV')-1] - pars['presPerFraction'])))
        obj.append(
            10000 * (1 / len(st.get_voxels_idx('PTV'))) * (cp.sum_squares(dO) + 10 * cp.sum_squares(dU)))

        ##Smoothing objective function

        # obj.append(1000 * (0.6 * cp.sum_squares(X @ w) + 0.4 * cp.sum_squares(Y @ w)))

        print('Objective done')
        print('Constraints Start')
        constraints = []
        # constraints += [wMean == cp.sum(w)/w.shape[0]]
        for i in range(len(clinical_constraints)):
            if 'max_hard_constraint_Gy' in clinical_constraints[i]:
                if clinical_constraints[i]['max_hard_constraint_Gy'] is not None:
                    org = clinical_constraints[i]['structNames']
                    if org != 'GTV':
                        constraints += [infMatrix[st.get_voxels_idx(org), :] @ w <= clinical_constraints[i][
                            'max_hard_constraint_Gy'] / num_fractions]
            if 'mean_hard_constraint_Gy' in clinical_constraints[i]:
                if clinical_constraints[i]['mean_hard_constraint_Gy'] is not None:
                    org = clinical_constraints[i]['structNames']
                    constraints += [
                        (1 / len(st.get_voxels_idx(org))) * (cp.sum(infMatrix[st.get_voxels_idx(org), :] @ w)) <=
                        clinical_constraints[i]['mean_hard_constraint_Gy'] / num_fractions]

        ##Step 1 and 2 constraint
        constraints += [infMatrix[st.get_voxels_idx('PTV'), :] @ w <= pres + dO]
        constraints += [infMatrix[st.get_voxels_idx('PTV'), :] @ w >= pres - dU]

        ##Smoothness Constraint
        # for b in range(len(self.beams.beams_dict['ID'])):
        #     self.beams.sort_beamlets()
        #     startB = myPlan['beams']['firstBeamlet'][b]
        #     endB = myPlan['beams']['endBeamlet'][b]
        #     constraints += [0.6 * cp.sum_squares(
        #         X[startB - 1:endB - 1, startB - 1:endB - 1] @ w[startB - 1:endB - 1]) + 0.4 * cp.sum_squares(
        #         Y[startB - 1:endB - 1, startB - 1:endB - 1] @ w[startB - 1:endB - 1]) <= 0.5]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)
        # Defining the constraints
        print('Problem loaded')
        prob.solve(solver=cp.MOSEK, verbose=True)
        print("optimal value with MOSEK:", prob.value)
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))
        return w.value

    @staticmethod
    def get_smoothness_matrix(beamReq):
        sRow = np.zeros((beamReq['endBeamlet'][-1], beamReq['endBeamlet'][-1]), dtype=int)
        sCol = np.zeros((beamReq['endBeamlet'][-1], beamReq['endBeamlet'][-1]), dtype=int)
        for b in range(len(beamReq['ID'])):
            map = beamReq['beamEyeViewBeamletMap'][b]

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
                while ((map[r, endCol] == -1) and (startCol <= endCol)):
                    endCol = endCol - 1

                for c in range(startCol, endCol + 1):
                    ind = map[r, c]
                    RN = map[r, c + 1]
                    if ind * RN > 0:
                        sRow[ind - 1, ind - 1] = int(1)
                        sRow[ind - 1, RN - 1] = int(-1)

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
                    if ind * DN > 0:
                        sCol[ind - 1, ind - 1] = int(1)
                        sCol[ind - 1, DN - 1] = int(-1)
        return sRow, sCol
