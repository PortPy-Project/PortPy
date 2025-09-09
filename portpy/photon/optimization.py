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
import numpy as np
import cvxpy as cp
from typing import List, TYPE_CHECKING, Union
import time
if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
from .clinical_criteria import ClinicalCriteria
from copy import deepcopy

class Optimization(object):
    """
    Optimization class for optimizing and creating the plan

    :param my_plan: object of class Plan
    :param inf_matrix: object of class InfluenceMatrix
    :param clinical_criteria: clinical criteria for which plan to be optimized
    :param opt_params: optimization parameters for modifying parameters of problem statement

    - **Attributes** ::

        :param obj: List containing individual objective function
        :param constraints: List containing individual constraints
        :param vars: Dictionary containing variable
        :Example
                dict = {"x": [...]}

    - **Methods** ::
        :create_cvxpy_problem(my_plan)
            Create cvxpy objective function and constraints and save them as a list

    """

    def __init__(self, my_plan: Plan, inf_matrix: InfluenceMatrix = None,
                 clinical_criteria: ClinicalCriteria = None,
                 opt_params: dict = None, vars: dict = None):
        # self.x = None
        self.my_plan = my_plan
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        self.inf_matrix = inf_matrix
        if clinical_criteria is None:
            clinical_criteria = my_plan.clinical_criteria
        self.clinical_criteria = clinical_criteria
        self.opt_params = opt_params
        # self.prescription_gy = opt_params['prescription_gy']
        self.obj = []
        self.constraints = []
        self.obj_value = None
        if vars is None:
            x = cp.Variable(inf_matrix.A.shape[1], pos=True, name='x')  # creating variable for beamlet intensity
            self.vars = {'x': x}
        else:
            self.vars = vars

    def create_cvxpy_problem(self):
        """
        It runs optimization to create optimal plan based upon clinical criteria

        This method constructs the components of the CVXPY optimization problem:
        - Populates :attr obj: with a list of individual objective terms
        - Populates :attr constraints: with dose-based and clinical constraints

        Note:
            This method does not return a CVXPY Problem object. Instead, it prepares the internal components
            used by :meth solve(): or for manual construction of a CVXPY Problem.

        :return:
        """

        # unpack data
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        opt_params = self.opt_params
        clinical_criteria = self.clinical_criteria
        x = self.vars['x']
        obj = self.obj
        constraints = self.constraints

        # self.prescription_gy = opt_params['prescription_gy']

        # get opt params for optimization
        obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
        opt_params_constraints = opt_params['constraints'] if 'constraints' in opt_params else []

        A = inf_matrix.A
        num_fractions = clinical_criteria.get_num_of_fractions()
        st = inf_matrix

        # Construct optimization problem

        # Generating objective functions
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dO))]
                    constraints += [A[st.get_opt_voxels_idx(struct), :] @ x <= dose_gy + dO]
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dU))]
                    constraints += [A[st.get_opt_voxels_idx(struct), :] @ x >= dose_gy - dU]
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(A[st.get_opt_voxels_idx(struct), :] @ x))]
            elif obj_funcs[i]['type'] == 'smoothness-quadratic':
                [Qx, Qy, num_rows, num_cols] = self.get_smoothness_matrix(inf_matrix.beamlets_dict)
                smoothness_X_weight = 0.6
                smoothness_Y_weight = 0.4
                obj += [obj_funcs[i]['weight'] * (smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) +
                                                  smoothness_Y_weight * (1 / num_rows) * cp.sum_squares(Qy @ x))]

        # Generating constraints
        constraint_def = deepcopy(clinical_criteria.get_criteria())  # get all constraints definition using clinical criteria

        # add/modify constraints definition if present in opt params
        for opt_constraint in opt_params_constraints:
            # add constraint
            param  = opt_constraint['parameters']
            if param['structure_name'] in my_plan.structures.get_structures():
                criterion_exist, criterion_ind = clinical_criteria.check_criterion_exists(opt_constraint, return_ind=True)
                if criterion_exist:
                    constraint_def[criterion_ind] = opt_constraint
                else:
                    constraint_def += [opt_constraint]

        d_max = np.infty * np.ones(A.shape[0]) # create a vector to avoid putting redundant max constraint on
        # duplicate voxels among structure

        # Adding max/mean constraints
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    if org != 'GTV' and org != 'CTV':
                        if org in my_plan.structures.get_structures():
                            if len(st.get_opt_voxels_idx(org)) == 0:
                                continue
                            voxels = st.get_opt_voxels_idx(org)
                            d_max[voxels] = np.minimum(d_max[voxels], limit / num_fractions)
                            # constraints += [A[st.get_opt_voxels_idx(org), :] @ x <= limit / num_fractions]
            elif constraint_def[i]['type'] == 'mean_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    # mean constraints using voxel weights
                    if org in my_plan.structures.get_structures():
                        if len(st.get_opt_voxels_idx(org)) == 0:
                            continue
                        fraction_of_vol_in_calc_box = my_plan.structures.get_fraction_of_vol_in_calc_box(org)
                        limit = limit/fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
                        constraints += [(1 / sum(st.get_opt_voxels_volume_cc(org))) *
                                        (cp.sum((cp.multiply(st.get_opt_voxels_volume_cc(org),
                                                             A[st.get_opt_voxels_idx(org), :] @ x))))
                                        <= limit / num_fractions]
        mask = np.isfinite(d_max)
        # Create index mask arrays
        indices = np.arange(len(mask))  # assumes mask is 1D and corresponds to voxel indices
        all_d_max_vox_ind = indices[mask]
        constraints += [A[all_d_max_vox_ind, :] @ x <= d_max[all_d_max_vox_ind]] # Add constraint for all d_max voxels at once
        print('Problem created')

    def add_max(self, struct: str, dose_gy: float):
        """
        Add max constraints to the problem

        :param struct: struct_name name
        :param dose_gy: dose in Gy per fraction.
        :return:
        """
        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']

        max_constraint = [A[st.get_opt_voxels_idx(struct), :] @ x <= dose_gy]
        self.add_constraints(max_constraint)

    def add_mean(self, struct: str, dose_gy: float):
        """
        Add mean constraints to the problem

        :param struct: struct_name name
        :param dose_gy: dose in Gy per fraction.
        :return:
        """
        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']

        mean_constraint = [(1 / sum(st.get_opt_voxels_volume_cc(struct))) *
                           (cp.sum((cp.multiply(st.get_opt_voxels_volume_cc(struct),
                                                A[st.get_opt_voxels_idx(struct),
                                                :] @ x)))) <= dose_gy]
        self.add_constraints(mean_constraint)

    def add_overdose_quad(self, struct: str, dose_gy: float, weight: float = 10000):
        """
        Add quadratic loss for the overdose voxels of the struct_name

        :param struct: struct_name name
        :param dose_gy: dose in Gy per fraction.
        :param weight: penalty/weight in the objective for overdose
        :return:
        """
        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']

        dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True, name='{}_overdose'.format(struct))
        obj = (1 / len(st.get_opt_voxels_idx(struct))) * (weight * cp.sum_squares(dO))
        self.add_objective(obj)
        self.add_constraints([A[st.get_opt_voxels_idx('PTV'), :] @ x <= dose_gy + dO])

    def add_underdose_quad(self, struct: str, dose_gy: float, weight: float = 100000):
        """
        Add quadratic loss for the underdose voxels of the struct_name

        :param struct: struct_name name
        :param dose_gy: dose in Gy per fraction.
        :param weight: penalty/weight in the objective for underdose
        :return:
        """
        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']

        dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True, name='{}_underdose'.format(struct))
        obj = (1 / len(st.get_opt_voxels_idx(struct))) * (weight * cp.sum_squares(dU))
        self.add_objective(obj)
        self.add_constraints([A[st.get_opt_voxels_idx('PTV'), :] @ x >= dose_gy - dU])

    def add_quad(self, struct: str = None, voxels: np.ndarray = None, weight: float = 10,
                 voxels_weight: np.ndarray = None):

        """
        Add quadratic objective to the optimization problem

        :param struct: struct_name for which quadratic loss is added to objective function
        :param voxels: Default to None. If set, quadratic loss will be added for the given voxels
        :param weight: Default to 10. penalty in the objective function for the given struct_name.
        :param voxels_weight: weight for each voxel in the objective function
        :return:
        """
        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']

        obj = 0
        if voxels is not None:
            if voxels_weight is None:
                raise Exception('Please input weight array for the input voxels')
            obj = (1 / A[voxels, :].shape[0]) * cp.sum_squares(
                cp.multiply(cp.sqrt(voxels_weight), A[voxels, :] @ x))
        if struct is not None:
            obj = (1 / len(st.get_opt_voxels_idx(struct))) * (
                    weight * cp.sum_squares(A[st.get_opt_voxels_idx(struct), :] @ x))
        self.add_objective(obj)

    def add_smoothness_quad(self, weight: int = 10, smoothness_X_weight: int = 0.6, smoothness_Y_weight: int = 0.4):
        """
        Add quadratic smoothness to the optimization problem

        :param weight: smoothness weight
        :param smoothness_X_weight: weight in X direction of MLC (parallel to MLC)
        :param smoothness_Y_weight: weight in Y direction of MLC (perpendicular to MLC)
        :return:
        """

        st = self.inf_matrix
        x = self.vars['x']

        [Qx, Qy, num_rows, num_cols] = self.get_smoothness_matrix(st.beamlets_dict)
        obj = weight * (
                smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) + smoothness_Y_weight * (1 / num_rows)
                * cp.sum_squares(Qy @ x))
        self.add_objective(obj)

    def add_constraints(self, constraints: list):
        """
        Add constraint to the constraint list of problem

        :param constraints: list of constraints
        :return:
        """
        self.constraints += constraints

    def add_objective(self, obj):
        """
        Add objective function to objective list of the problem

        :param obj: objective function expression using cvxpy
        :return:
        """
        if not isinstance(obj, list):
            obj = [obj]

        self.obj += obj

    def add_boo(self, num_beams: int):
        """
        Select optimal beams from set of beams using MIP

        :param num_beams: number of beams to be selected
        :return:
        """
        st = self.inf_matrix
        x = self.vars['x']

        #  Constraints for selecting beams
        # binary variable for selecting beams
        b = cp.Variable(len(st.beamlets_dict), boolean=True)

        constraints = []
        constraints += [cp.sum(b) <= num_beams]
        for i in range(len(st.beamlets_dict)):
            start_beamlet = st.beamlets_dict[i]['start_beamlet_idx']
            end_beamlet_idx = st.beamlets_dict[i]['end_beamlet_idx']
            M = 50  # upper bound on the beamlet intensity
            constraints += [x[start_beamlet:end_beamlet_idx] <= b[i] * M]

        self.add_constraints(constraints)

    def solve(self, return_cvxpy_prob=False, *args, **kwargs):
        """
        Return optimal solution and influence matrix associated with it in the form of dictionary
        If return_problem set to true, returns cvxpy problem instance

        :Example
                dict = {"optimal_fluence": [..],
                "inf_matrix": my_plan.inf_marix
                }

        :return: solution dictionary, cvxpy problem instance(optional)
        """

        problem = cp.Problem(cp.Minimize(cp.sum(self.obj)), constraints=self.constraints)
        print('Running Optimization..')
        t = time.time()
        # Check if 'solver' is passed in args
        solver = kwargs.get('solver', None)
        if solver and solver.lower() == 'mosek':
            try:
                problem.solve(*args, **kwargs)  # Attempt to solve with mosek
            except cp.error.SolverError as e:
                # Raise a custom error if MOSEK is not installed or available
                raise ImportError(
                    "MOSEK solver is not installed. You can obtain the MOSEK license file by applying using an .edu account. \n"
                    r"The license file should be placed in the directory C:\\Users\\username\\mosek."
                    "\n To use MOSEK, install it using: pip install portpy[mosek].\n"
                    "If a license is not available, you may try open-source or free solvers like SCS or ECOS. \n"
                    "Please refer to the CVXPy documentation for more information about its various solvers.\n"
                ) from e
        else:
            problem.solve(*args, **kwargs)  # Continue solving with other solvers
        elapsed = time.time() - t
        self.obj_value = problem.value
        print("Optimal value: %s" % problem.value)
        print("Elapsed time: {} seconds".format(elapsed))
        sol = {'optimal_intensity': self.vars['x'].value, 'inf_matrix': self.inf_matrix, 'obj_value': problem.value}
        if return_cvxpy_prob:
            return sol, problem
        else:
            return sol

    def get_sol(self) -> dict:
        """
        Return optimal solution and influence matrix associated with it in the form of dictionary

        :Example
                dict = {"optimal_fluence": [..],
                "inf_matrix": my_plan.inf_marix
                }

        :return: solution dictionary
        """
        return {'optimal_intensity': self.vars['x'].value, 'inf_matrix': self.inf_matrix}

    def add_dvh(self, dvh_constraint: list):

        A = self.inf_matrix.A
        st = self.inf_matrix
        x = self.vars['x']

        import pandas as pd
        df_dvh_criteria = pd.DataFrame()
        count = 0
        criteria = self.clinical_criteria.clinical_criteria_dict['criteria']
        for i in range(len(dvh_constraint)):
            if 'dose_volume' in dvh_constraint[i]['type']:
                limit_key = self.matching_keys(dvh_constraint[i]['constraints'], 'limit')
                if limit_key in dvh_constraint[i]['constraints']:
                    df_dvh_criteria.at[count, 'structure_name'] = dvh_constraint[i]['parameters']['structure_name']
                    df_dvh_criteria.at[count, 'dose_gy'] = dvh_constraint[i]['parameters']['dose_gy']

                    # getting max dose_1d for the same struct_name
                    max_dose_struct = 1000
                    for j in range(len(criteria)):
                        if 'max_dose' in criteria[j]['type']:
                            if 'limit_dose_gy' in criteria[j]['constraints']:
                                org = criteria[j]['parameters']['structure_name']
                                if org == dvh_constraint[i]['parameters']['structure_name']:
                                    max_dose_struct = criteria[j]['constraints']['limit_dose_gy']
                    df_dvh_criteria.at[count, 'M'] = max_dose_struct - dvh_constraint[i]['parameters']['dose_gy']
                    if 'perc' in limit_key:
                        df_dvh_criteria.at[count, 'vol_perc'] = dvh_constraint[i]['constraints'][limit_key]
                    count = count + 1

        # binary variable for dvh constraints
        b_dvh = cp.Variable(
            len(np.concatenate([st.get_opt_voxels_idx(org) for org in df_dvh_criteria.structure_name.to_list()])),
            boolean=True)

        start = 0
        constraints = []
        for i in range(len(df_dvh_criteria)):
            struct, limit, v, M = df_dvh_criteria.loc[i, 'structure_name'], df_dvh_criteria.loc[i, 'dose_gy'], \
                                  df_dvh_criteria.loc[i, 'vol_perc'], df_dvh_criteria.loc[i, 'M']
            end = start + len(st.get_opt_voxels_idx(struct))
            frac = self.my_plan.structures.get_fraction_of_vol_in_calc_box(struct)
            constraints += [
                A[st.get_opt_voxels_idx(struct), :] @ x <= limit / self.my_plan.get_num_of_fractions()
                + b_dvh[start:end] * M / self.my_plan.get_num_of_fractions()]
            constraints += [b_dvh @ st.get_opt_voxels_volume_cc(struct) <= (v / frac) / 100 * sum(
                st.get_opt_voxels_volume_cc(struct))]
            start = end
        self.add_constraints(constraints=constraints)

    @staticmethod
    def get_smoothness_matrix(beamReq: List[dict]) -> (np.ndarray, np.ndarray, int, int):
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
        sRow = np.zeros((beamReq[-1]['end_beamlet_idx'] + 1, beamReq[-1]['end_beamlet_idx'] + 1), dtype=int)
        sCol = np.zeros((beamReq[-1]['end_beamlet_idx'] + 1, beamReq[-1]['end_beamlet_idx'] + 1), dtype=int)
        num_rows = 0
        num_cols = 0
        for b in range(len(beamReq)):
            beam_map = beamReq[b]['beamlet_idx_2d_finest_grid']

            rowsNoRepeat = [0]
            for i in range(1, np.size(beam_map, 0)):
                if (beam_map[i, :] != beam_map[rowsNoRepeat[-1], :]).any():
                    rowsNoRepeat.append(i)
            colsNoRepeat = [0]
            for j in range(1, np.size(beam_map, 1)):
                if (beam_map[:, j] != beam_map[:, colsNoRepeat[-1]]).any():
                    colsNoRepeat.append(j)
            beam_map = beam_map[np.ix_(np.asarray(rowsNoRepeat), np.asarray(colsNoRepeat))]
            num_rows = num_rows + beam_map.shape[0]
            num_cols = num_cols + beam_map.shape[1]
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
        return sRow, sCol, num_rows, num_cols

    def create_cvxpy_problem_correction(self, d=None, delta=None):
        """
        It runs optimization to create optimal plan based upon clinical criteria
        :param d: cvxpy variable or expression containing the definition of dose. If not provided, it will use d = Ax by default
        :param delta: constant dose correction term to be used in optimization
        :return: cvxpy problem object

        """
        if delta is None:
            delta = np.zeros(self.inf_matrix.A.shape[0])
        # unpack data
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        opt_params = self.opt_params
        clinical_criteria = self.clinical_criteria
        x = self.vars['x']
        obj = self.obj
        constraints = self.constraints

        # self.prescription_gy = opt_params['prescription_gy']

        # get opt params for optimization
        obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
        opt_params_constraints = opt_params['constraints'] if 'constraints' in opt_params else []

        A = inf_matrix.A
        num_fractions = clinical_criteria.get_num_of_fractions()
        st = inf_matrix

        if d is None:
            d = cp.Variable(A.shape[0], pos=True, name='d')  # creating dummy variable for dose
            constraints += [d == A @ x]

        # Construct optimization problem

        # Generating objective functions
        print('Objective Start')
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dO = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dO))]
                    constraints += [d[st.get_opt_voxels_idx(struct)] + delta[st.get_opt_voxels_idx(struct)] <= dose_gy + dO]
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dU = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(dU))]
                    constraints += [d[st.get_opt_voxels_idx(struct)] + delta[st.get_opt_voxels_idx(struct)] >= dose_gy - dU]
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    obj += [(1 / len(st.get_opt_voxels_idx(struct))) * (obj_funcs[i]['weight'] * cp.sum_squares(d[st.get_opt_voxels_idx(struct)] + delta[st.get_opt_voxels_idx(struct)]))]
            elif obj_funcs[i]['type'] == 'smoothness-quadratic':
                [Qx, Qy, num_rows, num_cols] = self.get_smoothness_matrix(inf_matrix.beamlets_dict)
                smoothness_X_weight = 0.6
                smoothness_Y_weight = 0.4
                obj += [obj_funcs[i]['weight'] * (smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) +
                                                  smoothness_Y_weight * (1 / num_rows) * cp.sum_squares(Qy @ x))]

        print('Objective done')

        print('Constraints Start')

        constraint_def = deepcopy(clinical_criteria.get_criteria())  # get all constraints definition using clinical criteria

        # add/modify constraints definition if present in opt params
        for opt_constraint in opt_params_constraints:
            # add constraint
            param = opt_constraint['parameters']
            if param['structure_name'] in my_plan.structures.get_structures():
                criterion_exist, criterion_ind = clinical_criteria.check_criterion_exists(opt_constraint,
                                                                                          return_ind=True)
                if criterion_exist:
                    constraint_def[criterion_ind] = opt_constraint
                else:
                    constraint_def += [opt_constraint]

        # Adding max/mean constraints
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    if org != 'GTV' and org != 'CTV':
                        if org in my_plan.structures.get_structures():
                            if len(st.get_opt_voxels_idx(org)) == 0:
                                continue
                            constraints += [d[st.get_opt_voxels_idx(org)] + delta[st.get_opt_voxels_idx(org)] <= limit / num_fractions]
            elif constraint_def[i]['type'] == 'mean_dose':
                limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                if limit_key:
                    limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                    org = constraint_def[i]['parameters']['structure_name']
                    # mean constraints using voxel weights
                    if org in my_plan.structures.get_structures():
                        if len(st.get_opt_voxels_idx(org)) == 0:
                            continue
                        fraction_of_vol_in_calc_box = my_plan.structures.get_fraction_of_vol_in_calc_box(org)
                        limit = limit / fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
                        constraints += [(1 / sum(st.get_opt_voxels_volume_cc(org))) *
                                        (cp.sum((cp.multiply(st.get_opt_voxels_volume_cc(org),
                                                             d[st.get_opt_voxels_idx(org)] + delta[
                                                                 st.get_opt_voxels_idx(org)]))))
                                        <= limit / num_fractions]

        print('Constraints done')

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

    def get_num(self, string: Union[str, float]):
        if "prescription_gy" in str(string):
            prescription_gy = self.clinical_criteria.get_prescription()
            return eval(string)
        elif isinstance(string, float) or isinstance(string, int):
            return string
        else:
            raise Exception('Invalid constraint')

    def dose_to_gy(self, key, value):
        if "prescription_gy" in str(value):
            prescription_gy = self.clinical_criteria.get_prescription()
            return eval(value)
        elif 'gy' in key:
            return value
        elif 'perc' in key:
            return value*self.clinical_criteria.get_prescription()/100