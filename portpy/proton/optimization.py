from __future__ import annotations
import numpy as np
import cvxpy as cp
from typing import List, TYPE_CHECKING, Union
import time
if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
from portpy.photon import ClinicalCriteria
from portpy.photon import Optimization
from copy import deepcopy


class Optimization(Optimization):
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
        super().__init__(my_plan=my_plan, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria,
                         opt_params=opt_params, vars=vars)
        self.my_plan = my_plan

    def create_cvxpy_problem(self):
        """
        It runs optimization to create optimal plan based upon clinical criteria

        :return: cvxpy problem object

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

        print('Objective done')

        print('Constraints Start')

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
                            constraints += [A[st.get_opt_voxels_idx(org), :] @ x <= limit / num_fractions]
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


        print('Constraints done')

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
        problem.solve(*args, **kwargs)
        elapsed = time.time() - t
        self.obj_value = problem.value
        print("Optimal value: %s" % problem.value)
        print("Elapsed time: {} seconds".format(elapsed))
        sol = {'optimal_intensity': self.vars['x'].value, 'inf_matrix': self.inf_matrix}
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
            prescription_gy = self.prescription_gy
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