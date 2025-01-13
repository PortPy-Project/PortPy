from __future__ import annotations

from portpy.photon import Optimization
from typing import List, TYPE_CHECKING, Union
import time

if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
    from portpy.photon.vmat_scp.arcs import Arcs
from portpy.photon.clinical_criteria import ClinicalCriteria
import cvxpy as cp
import numpy as np
from copy import deepcopy


class VmatScpOptimization(Optimization):
    """
    Class for VMAT optimization using Sequential Convex Programming (SCP) method

    - **Attributes** ::
        :param my_plan: object of class Plan
        :param inf_matrix: object of class InfluenceMatrix
        :param clinical_criteria: object of class ClinicalCriteria
        :param opt_params: dictionary of vmat optimization parameters
        :param vars: dictionary of variables
        :param sol: Optional. solution to be passed for the optimization
        :param arcs: Optional. object of class Arcs

    :Example:
    >>> vmat_opt = VmatScpOptimization(my_plan=my_plan, inf_matrix=inf_matrix, clinical_criteria=clinical_criteria, opt_params=vmat_opt_params)
    >>> vmat_opt.run_sequential_cvx_algo(solver='MOSEK', verbose=True)

    - **Methods** ::
        :run_sequential_cvx_algo(solver: str, verbose: bool = False)
            Run Sequential Convex Programming algorithm for VMAT optimization
        :create_cvxpy_intermediate_problem()
            Creates cvxpy problem for ECHO
        :resolve_infeasibility_of_actual_solution(sol: dict, *args, **kwargs)
            Resolve infeasibility of the intermediate solution
        :create_cvxpy_actual_problem()
            Construct actual problem for optimizing MU
    """
    def __init__(self, my_plan: Plan, inf_matrix: InfluenceMatrix = None,
                 clinical_criteria: ClinicalCriteria = None,
                 opt_params: dict = None, vars: dict = None, sol=None, arcs: Arcs = None):
        # Call the constructor of the base class (Optimization) using super()
        super().__init__(my_plan=my_plan, inf_matrix=inf_matrix,
                         clinical_criteria=clinical_criteria,
                         opt_params=opt_params, vars=vars)
        # save previous solution if passed

        self.prev_sol = sol
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

    def create_cvxpy_intermediate_problem(self):
        """

        Creates intermediate cvxpy problem for optimizing interior and boundary beamlets
        :return: None

        """
        # unpack data
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        opt_params = self.opt_params
        clinical_criteria = self.clinical_criteria
        self.obj = []
        self.constraints = []
        obj = self.obj
        constraints = self.constraints
        x = self.vars['x']
        if self.outer_iteration == 0:
            self.create_cvx_params()
        inf_int, inf_bound_l, inf_bound_r = self.create_interior_and_boundary_inf_matrix()

        # get interior and boundary beamlets properties in matrix form
        map_int_v = self.cvxpy_params['map_int_v']
        min_bound_index_l = self.cvxpy_params['min_bound_index_l']
        not_empty_bound_l = self.cvxpy_params['not_empty_bound_l']
        current_leaf_pos_l = self.cvxpy_params['current_leaf_pos_l']
        card_bound_inds_l = self.cvxpy_params['card_bound_inds_l']
        min_bound_index_r = self.cvxpy_params['min_bound_index_r']
        not_empty_bound_r = self.cvxpy_params['not_empty_bound_r']
        current_leaf_pos_r = self.cvxpy_params['current_leaf_pos_r']
        card_bound_inds_r = self.cvxpy_params['card_bound_inds_r']
        map_adj_int = self.cvxpy_params['map_adj_int']
        map_adj_bound = self.cvxpy_params['map_adj_bound']
        total_rows = np.sum([arc['total_rows'] for arc in self.arcs.arcs_dict['arcs']])
        total_beams = np.sum([arc['num_beams'] for arc in self.arcs.arcs_dict['arcs']])

        # get opt params for optimization
        obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
        self.obj_funcs = obj_funcs
        opt_params_constraints = opt_params['constraints'] if 'constraints' in opt_params else []
        num_fractions = clinical_criteria.get_num_of_fractions()
        st = inf_matrix

        # Construct optimization problem
        # create variables
        leaf_pos_mu_l = cp.Variable(total_rows, pos=True)
        leaf_pos_mu_r = cp.Variable(total_rows, pos=True)
        int_v = cp.Variable(total_beams, pos=True)
        bound_v_l = cp.Variable(total_rows, pos=True)
        bound_v_r = cp.Variable(total_rows, pos=True)

        # save required variables in optimization object for future use
        self.vars['leaf_pos_mu_l'] = leaf_pos_mu_l
        self.vars['leaf_pos_mu_r'] = leaf_pos_mu_r
        self.vars['int_v'] = int_v
        self.vars['bound_v_l'] = bound_v_l
        self.vars['bound_v_r'] = bound_v_r

        # Generating objective functions
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    dO = 'dO_{}_{:.2f}'.format(struct, dose_gy)
                    voxels = st.get_opt_voxels_idx(struct)
                    voxels_vol_cc = st.get_opt_voxels_volume_cc(struct)
                    self.vars[dO] = cp.Variable(len(voxels), pos=True)
                    obj += [(1 / cp.sum(voxels_vol_cc)) * (obj_funcs[i]['weight']*cp.sum_squares(cp.multiply(cp.sqrt(voxels_vol_cc), self.vars[dO])))]
                    # inf_int is interior influence matrix, inf_bound_l is left boundary influence matrix, inf_bound_r is right boundary influence matrix
                    # int_v is interior beamlet intensity, bound_v_l is left boundary beamlet intensity, bound_v_r is right boundary beamlet intensity
                    # map_adj_int is mapping between interior variable and controlling MU for first and last beam due to inertia, map_adj_bound is similar
                    constraints += [inf_int[voxels, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[voxels, :] @ cp.multiply(bound_v_l, map_adj_bound)
                                    + inf_bound_r[voxels, :] @ cp.multiply(bound_v_r, map_adj_bound) <= dose_gy + self.vars[dO]]
                    print('Objective function type: {} , structure:{}, dose_gy:{}, weight:{} created..'.format(
                        obj_funcs[i]['type'], struct, dose_gy, obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    voxels = st.get_opt_voxels_idx(struct)
                    voxels_vol_cc = st.get_opt_voxels_volume_cc(struct)
                    dU = 'dU_{}_{:.2f}'.format(struct, dose_gy)
                    self.vars[dU] = cp.Variable(len(st.get_opt_voxels_idx(struct)), pos=True)
                    obj += [(1 / cp.sum(voxels_vol_cc)) * (obj_funcs[i]['weight']*cp.sum_squares(cp.multiply(cp.sqrt(voxels_vol_cc), self.vars[dU])))]
                    constraints += [inf_int[voxels, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[voxels, :] @ cp.multiply(
                        bound_v_l, map_adj_bound) + inf_bound_r[voxels, :] @ cp.multiply(bound_v_r, map_adj_bound) >= dose_gy - self.vars[dU]]
                    print('Objective function type: {} , structure:{}, dose_gy:{}, weight:{} created..'.format(
                        obj_funcs[i]['type'], struct, dose_gy, obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in my_plan.structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(st.get_opt_voxels_idx(struct)) == 0:
                        continue
                    voxels = st.get_opt_voxels_idx(struct)
                    voxels_vol_cc = st.get_opt_voxels_volume_cc(struct)
                    obj += [(1 / cp.sum(voxels_vol_cc)) * (obj_funcs[i]['weight'] * cp.sum_squares(cp.multiply(cp.sqrt(voxels_vol_cc), inf_int[voxels, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[voxels, :] @ cp.multiply(
                            bound_v_l, map_adj_bound) + inf_bound_r[voxels, :] @ cp.multiply(bound_v_r, map_adj_bound))))]
                    print('Objective function type: {}, structure:{}, weight:{} created..'.format(obj_funcs[i]['type'], struct, obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'aperture_regularity_quadratic':
                apt_reg_m = self.cvxpy_params['apt_reg_m']
                card_ar = self.cvxpy_params['card_ar']
                weight = obj_funcs[i]['weight'] * (my_plan.get_prescription() / my_plan.get_num_of_fractions())
                obj += [weight / card_ar * (cp.sum_squares(apt_reg_m @ leaf_pos_mu_l) +
                    cp.sum_squares(apt_reg_m @ leaf_pos_mu_r))]
                print('Objective function type: {}, weight:{} created..'.format(obj_funcs[i]['type'],
                                                                                obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'aperture_similarity_quadratic':
                apt_sim_m = self.cvxpy_params['apt_sim_m']
                card_as = self.cvxpy_params['card_as']
                weight = obj_funcs[i]['weight'] * (my_plan.get_prescription() / my_plan.get_num_of_fractions())
                obj += [weight / card_as * (cp.sum_squares(apt_sim_m @ leaf_pos_mu_l) +
                    cp.sum_squares(apt_sim_m @ leaf_pos_mu_r))]
                print('Objective function type: {}, weight:{} created..'.format(obj_funcs[i]['type'],
                                                                                obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'similar_mu_linear':
                similar_mu_obj = []
                index_stop = 0
                index_start = 0

                for arc in self.arcs.arcs_dict['arcs']:
                    index_stop += arc['num_beams']
                    for j in range(index_start, index_stop - 1):
                        similar_mu_obj += [obj_funcs[i]['weight'] * cp.abs(int_v[j] - int_v[j + 1])]
                    index_start += arc['num_beams']
                obj += [cp.sum(similar_mu_obj)]
                print('Objective function type: {}, weight:{} created..'.format(obj_funcs[i]['type'], obj_funcs[i]['weight']))
        # Create convex leaf positions
        constraints += [
            leaf_pos_mu_l == cp.multiply(cp.multiply(1 - not_empty_bound_l, current_leaf_pos_l), int_v[map_int_v]) +
            cp.multiply(cp.multiply(not_empty_bound_l, min_bound_index_l), int_v[map_int_v])
            + cp.multiply((int_v[map_int_v] - bound_v_l), card_bound_inds_l)]
        constraints += [
            leaf_pos_mu_r == cp.multiply(cp.multiply(1 - not_empty_bound_r, current_leaf_pos_r), int_v[map_int_v]) +
            cp.multiply(cp.multiply(not_empty_bound_r, min_bound_index_r), int_v[map_int_v])
            + cp.multiply(bound_v_r, card_bound_inds_r)]
        # generic constraints for relation between interior and boundary beamlets
        constraints += [leaf_pos_mu_r - leaf_pos_mu_l >= int_v[map_int_v]]
        constraints += [int_v >= self.vmat_params['mu_min']]
        constraints += [bound_v_l <= int_v[map_int_v]]
        constraints += [bound_v_r <= int_v[map_int_v]]

        # minimum dyanmic leaf gap constraint
        if 'minimum_dynamic_leaf_gap_mm' in self.vmat_params:
            min_leaf_gap_beamlet = self.vmat_params['minimum_dynamic_leaf_gap_mm'] / my_plan.beams.get_beamlet_width() * 1.01
            constraints += [leaf_pos_mu_r - leaf_pos_mu_l >= int_v[map_int_v] * min_leaf_gap_beamlet]

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

        self.constraint_def = constraint_def

        # imrt version
        # Adding max/mean constraints
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                org = constraint_def[i]['parameters']['structure_name']
                if org != 'GTV' and org != 'CTV':
                    if org in my_plan.structures.get_structures():
                        if len(st.get_opt_voxels_idx(org)) == 0:
                            continue
                        limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                        voxels = st.get_opt_voxels_idx(org)
                        if limit_key:
                            limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                            constraints += [inf_int[voxels, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[voxels, :] @ cp.multiply(
                                bound_v_l, map_adj_bound) + inf_bound_r[voxels, :] @ cp.multiply(bound_v_r, map_adj_bound) <= limit / num_fractions]
                            print('Constraint type: {}, structure:{}, limit_gy:{} created..'.format(constraint_def[i]['type'], org, limit / num_fractions))
            elif constraint_def[i]['type'] == 'mean_dose':
                org = constraint_def[i]['parameters']['structure_name']
                if org in my_plan.structures.get_structures():
                    if len(st.get_opt_voxels_idx(org)) == 0:
                        continue
                    limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                    voxels = st.get_opt_voxels_idx(org)
                    voxels_cc = st.get_opt_voxels_volume_cc(org)
                    fraction_of_vol_in_calc_box = my_plan.structures.get_fraction_of_vol_in_calc_box(org)
                    if limit_key:
                        limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                        limit = limit / fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
                        constraints += [(1 / sum(voxels_cc)) * (cp.sum((cp.multiply(voxels_cc, inf_int[voxels, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[voxels, :] @ cp.multiply(
                                bound_v_l, map_adj_bound) + inf_bound_r[voxels, :] @ cp.multiply(bound_v_r, map_adj_bound))))) <= limit / num_fractions]
                        print('Constraint type: {}, structure:{}, limit_gy:{} created..'.format(constraint_def[i]['type'], org, limit / num_fractions))

    def resolve_infeasibility_of_actual_solution(self, sol: dict, *args, **kwargs):
        """
        Resolve infeasibility of the intermediate solution
        :param sol: solution to be checked for feasibility

        returns sol: actual feasible solution
        """
        dev_max_dose = 0
        dev_mean_dose = 0
        num_fractions = self.my_plan.get_num_of_fractions()
        inf_matrix = self.inf_matrix
        constraint_def = self.constraint_def

        # check if infeasible
        if self.vmat_params['step_size_f'] > 1:
            for i in range(len(constraint_def)):
                if constraint_def[i]['type'] == 'max_dose':
                    org = constraint_def[i]['parameters']['structure_name']
                    if org != 'GTV' and org != 'CTV':
                        if org in self.my_plan.structures.get_structures():
                            voxels = inf_matrix.get_opt_voxels_idx(org)
                            if len(voxels) == 0:
                                continue
                            limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                            if limit_key:
                                limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                                # limit = self.get_num(constraint_def[i]['constraints']['limit_dose_gy'])
                                test1 = np.max(sol['act_dose_v'][voxels]) - limit / num_fractions
                                if test1 > 0:
                                    print("Violating max constraint for structure {}".format(org))
                                dev_max_dose = np.maximum(dev_max_dose, np.max(sol['act_dose_v'][voxels]) - limit / num_fractions)

                elif constraint_def[i]['type'] == 'mean_dose':
                    org = constraint_def[i]['parameters']['structure_name']
                    if org in self.my_plan.structures.get_structures():
                        voxels = inf_matrix.get_opt_voxels_idx(org)
                        if len(voxels) == 0:
                            continue
                        limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                        if limit_key:
                            limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                            voxels_vol = inf_matrix.get_opt_voxels_volume_cc(org)
                            fraction_of_vol_in_calc_box = self.my_plan.structures.get_fraction_of_vol_in_calc_box(org)
                            limit = limit / fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
                            dev_mean_dose = np.maximum(dev_mean_dose, (1 / sum(voxels_vol) * np.sum(np.multiply(voxels_vol, sol['act_dose_v'][voxels]))) - limit / num_fractions)

        # resolve infeasibility
        if dev_max_dose > self.vmat_params['dose_threshold'] or dev_mean_dose > self.vmat_params['dose_threshold']:
            print('Solving actual problem correction')
            self.create_cvxpy_actual_problem()
            sol = self.solve(actual_sol_correction=True, sol=sol, *args, **kwargs)
            beam_mu = sol['beam_mu']

            beams_so_far = 0
            w = np.zeros(inf_matrix.A.shape[1])
            arcs = self.arcs.arcs_dict['arcs']
            w_beamlet_act_corr = self.cvxpy_params['w_beamlet_act_corr']
            for a, arc in enumerate(arcs):
                num_beams = arc['num_beams']

                for b, beam in enumerate(arc['vmat_opt']):
                    from_ = beam['start_beamlet_idx']
                    to_ = beam['end_beamlet_idx']
                    w[from_:to_ + 1] = w_beamlet_act_corr[from_:to_ + 1] * beam_mu[beams_so_far + b]

                    beam['int_v'] = beam_mu[beams_so_far + b]
                arc['w_beamlet_act'] = w[arc['start_beamlet_idx']:arc['end_beamlet_idx'] + 1]

                beams_so_far += num_beams
            sol = self.arcs.calculate_dose(inf_matrix=self.inf_matrix, sol=sol, vmat_params=self.vmat_params)
            sol = self.calc_actual_objective_value(sol=sol, actual_sol_correction=True)
        return sol

    def create_cvxpy_actual_problem(self):
        """
        Construct actual problem for optimizing MU
        """
        # Construct actual solution correction problem

        #unpack data
        inf_apt = self.create_cvx_params(actual_sol_correction=True)
        total_beams = np.sum([arc['num_beams'] for arc in self.arcs.arcs_dict['arcs']])
        inf_matrix = self.inf_matrix
        structures = self.my_plan.structures
        obj_funcs = self.obj_funcs
        map_int_v = self.cvxpy_params['map_int_v']
        num_fractions = self.my_plan.get_num_of_fractions()
        pres_per_fraction = self.my_plan.get_prescription() / num_fractions
        fixed_leaf_pos_l = self.cvxpy_params['fixed_leaf_pos_l']
        fixed_leaf_pos_r = self.cvxpy_params['fixed_leaf_pos_r']

        # create variables and reference them
        beam_mu = cp.Variable(total_beams, pos=True)
        self.vars['beam_mu'] = beam_mu

        self.obj_actual = []  # empty if there is any other actual objectives and constraints
        self.constraints_actual = []
        obj_actual = self.obj_actual
        constraints_actual = self.constraints_actual
        # create objectives and constraints
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_vol_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    dO = 'actual_dO_{}_{:.2f}'.format(struct, dose_gy)
                    self.vars[dO] = cp.Variable(len(voxels), pos=True)
                    obj_actual += [(1 / cp.sum(voxels_vol_cc)) * (obj_funcs[i]['weight'] * cp.sum_squares(cp.multiply(cp.sqrt(voxels_vol_cc), self.vars[dO])))]
                    constraints_actual += [inf_apt[voxels, :] @ beam_mu <= dose_gy + self.vars[dO]]
                    print('Actual objective function type: {} , structure:{}, dose_gy:{}, weight:{} created..'.format(obj_funcs[i]['type'], struct, dose_gy, obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(struct)) == 0:
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_vol_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    dU = 'actual_dU_{}_{:.2f}'.format(struct, dose_gy)
                    self.vars[dU] = cp.Variable(len(voxels), pos=True)
                    obj_actual += [(1 / cp.sum(voxels_vol_cc)) * (obj_funcs[i]['weight'] * cp.sum_squares(cp.multiply(cp.sqrt(voxels_vol_cc), self.vars[dU])))]
                    constraints_actual += [inf_apt[voxels, :] @ beam_mu >= dose_gy - self.vars[dU]]
                    print('Actual objective function type: {} , structure:{}, dose_gy:{}, weight:{} created..'.format(obj_funcs[i]['type'], struct, dose_gy, obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(struct)) == 0:
                        continue
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_vol_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    obj_actual += [(1 / cp.sum(voxels_vol_cc)) * (obj_funcs[i]['weight'] * cp.sum_squares(cp.multiply(cp.sqrt(voxels_vol_cc), inf_apt[voxels, :] @ beam_mu)))]
                    print('Actual objective function type: {}, structure:{}, weight:{} created..'.format(obj_funcs[i]['type'],
                                                                                                  struct, obj_funcs[i][
                                                                                                      'weight']))
            elif obj_funcs[i]['type'] == 'aperture_regularity_quadratic':
                apt_reg_m = self.cvxpy_params['apt_reg_m']
                card_ar = self.cvxpy_params['card_ar']
                weight = obj_funcs[i]['weight'] * pres_per_fraction
                obj_actual += [weight / card_ar * (cp.sum_squares(apt_reg_m @ cp.multiply(fixed_leaf_pos_l, beam_mu[map_int_v])) +
                    cp.sum_squares(apt_reg_m @ cp.multiply(fixed_leaf_pos_r, beam_mu[map_int_v])))]
                print('Actual objective function type: {}, weight:{} created..'.format(obj_funcs[i]['type'],
                                                                                obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'aperture_similarity_quadratic':
                apt_sim_m = self.cvxpy_params['apt_sim_m']
                card_as = self.cvxpy_params['card_as']
                weight = obj_funcs[i]['weight'] * pres_per_fraction
                obj_actual += [weight / card_as * (cp.sum_squares(apt_sim_m @ cp.multiply(fixed_leaf_pos_l, beam_mu[map_int_v])) +
                    cp.sum_squares(apt_sim_m @ cp.multiply(fixed_leaf_pos_r, beam_mu[map_int_v])))]
                print('Actual objective function type: {}, weight:{} created..'.format(obj_funcs[i]['type'],
                                                                                obj_funcs[i]['weight']))
            elif obj_funcs[i]['type'] == 'similar_mu_linear':
                similar_mu_obj = []
                index_stop = 0
                index_start = 0
                print('Objective for similar MU between consecutive control points added..')
                for arc in self.arcs.arcs_dict['arcs']:
                    index_stop += arc['num_beams']
                    for j in range(index_start, index_stop - 1):
                        similar_mu_obj += [obj_funcs[i]['weight'] * cp.abs(beam_mu[j] - beam_mu[j + 1])]
                    index_start += arc['num_beams']
                obj_actual += [cp.sum(similar_mu_obj)]

        constraints_actual += [beam_mu >= self.vmat_params['mu_min']]

        # Adding max/mean constraints
        constraint_def = self.constraint_def
        for i in range(len(constraint_def)):
            if constraint_def[i]['type'] == 'max_dose':
                org = constraint_def[i]['parameters']['structure_name']
                if org != 'GTV' and org != 'CTV':
                    if org in structures.get_structures():
                        voxels = inf_matrix.get_opt_voxels_idx(org)
                        if len(voxels) == 0:
                            continue
                        limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                        if limit_key:
                            limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                            constraints_actual += [inf_apt[voxels, :] @ beam_mu <= limit / num_fractions]
                            print('Constraint type: {}, structure:{}, limit_gy:{} created..'.format(constraint_def[i]['type'], org, limit / num_fractions))
            elif constraint_def[i]['type'] == 'mean_dose':
                org = constraint_def[i]['parameters']['structure_name']
                # mean constraints using voxel weights
                if org in structures.get_structures():
                    voxels = inf_matrix.get_opt_voxels_idx(org)
                    if len(voxels) == 0:
                        continue
                    limit_key = self.matching_keys(constraint_def[i]['constraints'], 'limit')
                    if limit_key:
                        limit = self.dose_to_gy(limit_key, constraint_def[i]['constraints'][limit_key])
                        voxels_vol = inf_matrix.get_opt_voxels_volume_cc(org)
                        fraction_of_vol_in_calc_box = structures.get_fraction_of_vol_in_calc_box(org)
                        limit = limit / fraction_of_vol_in_calc_box  # modify limit due to fraction of volume receiving no dose
                        constraints_actual += [(1 / sum(voxels_vol)) * (cp.sum((cp.multiply(voxels_vol, inf_apt[voxels, :] @ beam_mu)))) <= limit / num_fractions]
                        print('Constraint type: {}, limit_gy:{} created..'.format(constraint_def[i]['type'], limit / num_fractions))
        return

    def create_interior_and_boundary_inf_matrix(self):
        """
        Create influence matrix based on interior and boundary beamlets

        :return: inf_int, inf_bound_l, inf_bound_r
        """
        print("Modifying influence matrix for boundary and interior beamlets. This process may take sometime..")
        A = self.inf_matrix.A
        arcs = self.arcs.arcs_dict['arcs']
        total_beams = sum([arc['num_beams'] for arc in arcs])
        total_rows = sum([arc['total_rows'] for arc in arcs])
        num_points = A.shape[0]
        inf_bound_l = np.zeros((num_points, total_rows))
        inf_bound_r = np.zeros((num_points, total_rows))
        inf_int = np.zeros((num_points, total_beams))

        cvxpy_params = self.cvxpy_params
        cvxpy_params['card_int_inds'] = np.zeros(total_beams, dtype=int)
        cvxpy_params['card_bound_inds_l'] = np.zeros(total_rows, dtype=int)
        cvxpy_params['card_bound_inds_r'] = np.zeros(total_rows, dtype=int)
        cvxpy_params['not_empty_bound_l'] = np.zeros(total_rows, dtype=int)
        cvxpy_params['not_empty_bound_r'] = np.zeros(total_rows, dtype=int)
        cvxpy_params['current_leaf_pos_l'] = np.zeros(total_rows, dtype=int)
        cvxpy_params['current_leaf_pos_r'] = np.zeros(total_rows, dtype=int)
        cvxpy_params['min_bound_index_l'] = np.zeros(total_rows, dtype=int)
        cvxpy_params['min_bound_index_r'] = np.zeros(total_rows, dtype=int)

        row_so_far = 0
        for a, arc in enumerate(arcs):
            vmat = arc['vmat_opt']
            num_beams = arc['num_beams']

            for b in range(num_beams):
                bound_ind_l = vmat[b]['bound_ind_left']
                bound_ind_r = vmat[b]['bound_ind_right']
                num_rows = vmat[b]['num_rows']
                reduced_2d_grid = vmat[b]['reduced_2d_grid']
                inf_int[:, sum([arc['num_beams'] for arc in arcs[:a]]) + b] = np.sum(A[:, vmat[b]['int_ind']].T, axis=0)
                for r in range(num_rows):
                    cvxpy_params['current_leaf_pos_l'][row_so_far] = vmat[b]['leaf_pos_left'][r] + 1
                    cvxpy_params['current_leaf_pos_r'][row_so_far] = vmat[b]['leaf_pos_right'][r]
                    if bound_ind_l[r]:
                        cvxpy_params['card_bound_inds_l'][row_so_far] = len(bound_ind_l[r])
                        col = np.argwhere(reduced_2d_grid == bound_ind_l[r][0])[0][1]  # get column of first boundary beamlet
                        cvxpy_params['min_bound_index_l'][row_so_far] = col
                        cvxpy_params['not_empty_bound_l'][row_so_far] = 1
                        inf_bound_l[:, row_so_far] = np.sum(A[:, vmat[b]['bound_ind_left'][r]].T, axis=0)
                    if bound_ind_r[r]:
                        cvxpy_params['card_bound_inds_r'][row_so_far] = len(bound_ind_r[r])
                        col = np.argwhere(reduced_2d_grid == bound_ind_r[r][0])[0][1]
                        cvxpy_params['min_bound_index_r'][row_so_far] = col
                        cvxpy_params['not_empty_bound_r'][row_so_far] = 1
                        inf_bound_r[:, row_so_far] = np.sum(A[:, vmat[b]['bound_ind_right'][r]].T, axis=0)
                    row_so_far = row_so_far + 1
        return inf_int, inf_bound_l, inf_bound_r

    def create_cvx_params(self, actual_sol_correction: bool = False):

        """
        Create cvxpy related matrices for objective function and constraints


        """
        if not actual_sol_correction:
            arcs = self.arcs.arcs_dict['arcs']
            cvxpy_params = self.cvxpy_params
            total_beams = np.sum([arc['num_beams'] for arc in arcs])
            total_rows = np.sum([arc['total_rows'] for arc in arcs])
            map_int_v = np.zeros(total_rows, dtype=int)
            apt_reg_m = np.zeros((total_rows, total_rows), dtype=int)
            apt_sim_m = np.zeros((total_rows, total_rows), dtype=int)
            row_so_far = 0
            beam_so_far = 0
            card_ar = 0
            for i, arc in enumerate(arcs):
                for j, beam in enumerate(arc['vmat_opt']):
                    for r in range(beam['num_rows']):
                        curr_row = row_so_far + r
                        map_int_v[curr_row] = beam_so_far + j
                        if r <= beam['num_rows'] - 2:
                            apt_reg_m[curr_row, curr_row] = 1
                            apt_reg_m[curr_row, curr_row + 1] = -1
                            card_ar = card_ar + 1
                    row_so_far = row_so_far + beam['num_rows']
                beam_so_far = beam_so_far + len(arc['vmat_opt'])

            cvxpy_params['apt_reg_m'] = apt_reg_m
            cvxpy_params['card_ar'] = card_ar
            cvxpy_params['map_int_v'] = map_int_v
            # aperture similarity
            matrix_row_ind = 0
            card_as = 0
            for i, arc in enumerate(arcs):
                for j, beam in enumerate(arc['vmat_opt']):
                    if j < len(arc['vmat_opt']) - 1:
                        next_beam = arc['vmat_opt'][j + 1]
                        curr_leaf_pairs = np.arange(beam['start_leaf_pair'], beam['end_leaf_pair'] - 1, -1)
                        next_leaf_pairs = np.arange(next_beam['start_leaf_pair'], next_beam['end_leaf_pair'] - 1,
                                                    -1)
                        current_index = 0
                        next_index = 0
                        while current_index < beam['num_rows'] and next_index < next_beam['num_rows']:
                            if curr_leaf_pairs[current_index] == next_leaf_pairs[next_index]:
                                apt_sim_m[matrix_row_ind + current_index, matrix_row_ind + current_index] = 1
                                next_beam_leaf_ind = matrix_row_ind + beam['num_rows'] + next_index
                                apt_sim_m[matrix_row_ind + current_index, next_beam_leaf_ind] = -1
                                current_index = current_index + 1
                                next_index = next_index + 1
                                card_as = card_as + 1
                            elif curr_leaf_pairs[current_index] > next_leaf_pairs[next_index]:
                                current_index = current_index + 1
                            elif curr_leaf_pairs[current_index] < next_leaf_pairs[next_index]:
                                next_index = next_index + 1
                        matrix_row_ind = matrix_row_ind + beam['num_rows']
                    else:
                        matrix_row_ind = matrix_row_ind + beam['num_rows']
            cvxpy_params['apt_sim_m'] = apt_sim_m
            cvxpy_params['card_as'] = card_as
            map_adj_int = np.ones(total_beams)
            map_adj_bound = np.ones(total_rows)

            vmat_params = self.vmat_params
            row_so_far = 0
            beam_so_far = 0
            for i, arc in enumerate(arcs):
                for j, beam in enumerate(arc['vmat_opt']):
                    if j == 0:
                        map_adj_int[beam_so_far] = vmat_params['first_beam_adj']
                        map_adj_bound[row_so_far:row_so_far + beam['num_rows']] = vmat_params['first_beam_adj']
                    elif j == 1:
                        map_adj_int[beam_so_far] = vmat_params['second_beam_adj']  # hard coded for now. Change it for 2nd and last beam
                        map_adj_bound[row_so_far:row_so_far + beam['num_rows']] = vmat_params['second_beam_adj']
                    # elif j == arc['num_beams'] - 1:
                    #     map_adj_int[beam_so_far] = vmat_params['last_beam_adj']
                    #     map_adj_bound[row_so_far:row_so_far + beam['num_rows']] = vmat_params['last_beam_adj']
                    row_so_far = row_so_far + beam['num_rows']
                    beam_so_far = beam_so_far + 1
            cvxpy_params['map_adj_int'] = map_adj_int
            cvxpy_params['map_adj_bound'] = map_adj_bound
        else:
            inf_matrix = self.inf_matrix
            A = inf_matrix.A
            arcs = self.arcs.arcs_dict['arcs']
            num_beamlets_so_far = 0
            fixed_leaf_pos_l = []
            fixed_leaf_pos_r = []
            w_beamlet_act_corr = np.zeros(A.shape[1])
            total_beams = sum([arc['num_beams'] for arc in arcs])
            inf_apt = np.zeros((A.shape[0], total_beams))
            adj0 = self.vmat_params['first_beam_adj']
            adj1 = self.vmat_params['second_beam_adj']
            adj2 = self.vmat_params['last_beam_adj']

            for a, arc in enumerate(arcs):
                num_beamlets = arc['end_beamlet_idx'] - arc['start_beamlet_idx'] + 1

                for b, beam in enumerate(arc['vmat_opt']):
                    range_ = np.arange(beam['start_beamlet_idx'] - num_beamlets_so_far,
                                       beam['end_beamlet_idx'] - num_beamlets_so_far + 1)
                    range2 = np.arange(beam['start_beamlet_idx'], beam['end_beamlet_idx'] + 1)

                    if beam['int_v'] > 0:
                        w_beamlet_act_corr[range2] = arc['w_beamlet_act'][range_] / beam['int_v']

                    for r in range(beam['num_rows']):
                        fixed_leaf_pos_l.append(beam['cont_leaf_pos_in_beamlet'][r, 0])
                        fixed_leaf_pos_r.append(beam['cont_leaf_pos_in_beamlet'][r, 1])

                    inf_apt[:, sum([arc['num_beams'] for arc in arcs[:a]]) + b] = A[:, range2] @ w_beamlet_act_corr[
                        range2] * ((b == 0) * adj0 + (b == 1) * adj1 + (1 < b <= (arc['num_beams'] - 1))*1)

                num_beamlets_so_far += num_beamlets

            self.cvxpy_params['fixed_leaf_pos_l'] = np.array(fixed_leaf_pos_l)
            self.cvxpy_params['fixed_leaf_pos_r'] = np.array(fixed_leaf_pos_r)
            self.cvxpy_params['w_beamlet_act_corr'] = w_beamlet_act_corr
            return inf_apt

    def calc_actual_objective_value(self, sol: dict, actual_sol_correction: bool = False):
        """
        Calculate actual objective function value using actual solution

        """
        # unpack data and optimization problems
        obj_funcs = self.obj_funcs
        structures = self.my_plan.structures
        inf_matrix = self.my_plan.inf_matrix
        num_fractions = self.my_plan.get_num_of_fractions()
        sol['overdose_obj'] = 0
        sol['underdose_obj'] = 0
        sol['quadratic_obj'] = 0
        sol['overdose_obj_norm'] = 0
        sol['underdose_obj_norm'] = 0
        sol['aperture_regularity_actual_obj_value'] = 0
        sol['aperture_similarity_actual_obj_value'] = 0
        sol['similar_mu_obj_value'] = 0
        obj_ind = 0
        # check if we have smooth objective
        for i in range(len(obj_funcs)):
            if obj_funcs[i]['type'] == 'quadratic-overdose':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(
                            struct)) == 0:  # check if there are any opt voxels for the structure
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    obj_value = (1 / np.sum(voxels_cc)) * obj_funcs[i]['weight'] * np.sum(voxels_cc *
                                                                                              (np.maximum(0, (sol['act_dose_v'][voxels] - dose_gy)) ** 2))
                    sol['overdose_obj_norm'] += obj_value/obj_funcs[i]['weight']
                    sol['overdose_obj'] += obj_value
                    obj_ind = obj_ind + 1
            elif obj_funcs[i]['type'] == 'quadratic-underdose':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(struct)) == 0:
                        continue
                    key = self.matching_keys(obj_funcs[i], 'dose')
                    dose_gy = self.dose_to_gy(key, obj_funcs[i][key]) / num_fractions
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    obj_value = (1 / np.sum(voxels_cc)) * obj_funcs[i]['weight'] * np.sum(voxels_cc *
                                                       (np.maximum(0, (dose_gy - sol['act_dose_v'][voxels])) ** 2))
                    sol['underdose_obj_norm'] += obj_value / obj_funcs[i]['weight']
                    sol['underdose_obj'] += obj_value
                    obj_ind = obj_ind + 1
            elif obj_funcs[i]['type'] == 'quadratic':
                if obj_funcs[i]['structure_name'] in structures.get_structures():
                    struct = obj_funcs[i]['structure_name']
                    if len(inf_matrix.get_opt_voxels_idx(struct)) == 0:
                        continue
                    voxels = inf_matrix.get_opt_voxels_idx(struct)
                    voxels_cc = inf_matrix.get_opt_voxels_volume_cc(struct)
                    sol['quadratic_obj'] += (1 / np.sum(voxels_cc)) * obj_funcs[i]['weight'] * np.sum(voxels_cc * (sol['act_dose_v'][voxels] ** 2))
                    obj_ind = obj_ind + 1
            elif obj_funcs[i]['type'] == 'aperture_regularity_quadratic':
                if actual_sol_correction:
                    sol['aperture_regularity_actual_obj_value'] += self.obj_actual[obj_ind].value
                else:
                    sol['aperture_regularity_actual_obj_value'] += self.obj[obj_ind].value
                obj_ind = obj_ind + 1
            elif obj_funcs[i]['type'] == 'aperture_similarity_quadratic':
                if actual_sol_correction:
                    sol['aperture_similarity_actual_obj_value'] += self.obj_actual[obj_ind].value
                else:
                    sol['aperture_similarity_actual_obj_value'] += self.obj[obj_ind].value
                obj_ind = obj_ind + 1
            elif obj_funcs[i]['type'] == 'similar_mu_linear':
                if actual_sol_correction:
                    sol['similar_mu_obj_value'] += self.obj_actual[obj_ind].value
                else:
                    sol['similar_mu_obj_value'] += self.obj[obj_ind].value
                obj_ind = obj_ind + 1


        sol['actual_obj_value'] = np.round((sol['overdose_obj'] + sol['underdose_obj'] + sol['quadratic_obj'] +
                                            sol['aperture_regularity_actual_obj_value'] +
                                            sol['aperture_similarity_actual_obj_value']) + sol['similar_mu_obj_value'], 4)
        return sol

    def run_sequential_cvx_algo(self, *args, **kwargs):
        """
        Returns sol and convergence of the sequential convex algorithm for optimizing the plan.
        Solver parameters can be passed in args.

        """
        # running scp algorithm:
        inner_iteration = int(0)
        best_obj_value = 0
        vmat_params = self.vmat_params
        self.arcs.get_initial_leaf_pos(initial_leaf_pos=vmat_params['initial_leaf_pos'])
        sol_convergence = []
        while True:

            self.arcs.gen_interior_and_boundary_beamlets(forward_backward=vmat_params['forward_backward'], step_size_f=vmat_params['step_size_f'], step_size_b=vmat_params['step_size_b'])
            self.create_cvxpy_intermediate_problem()
            sol = self.solve(*args, **kwargs)
            sol_convergence.append(sol)

            # post processing
            self.arcs.calc_actual_from_intermediate_sol(sol)
            sol = self.arcs.calculate_dose(inf_matrix=self.inf_matrix, sol=sol, vmat_params=vmat_params, best_plan=False)
            sol = self.calc_actual_objective_value(sol)

            sol = self.resolve_infeasibility_of_actual_solution(sol=sol, *args, **kwargs)

            if inner_iteration == 0:

                intial_step_size = int(np.maximum(3, np.ceil(self.arcs.get_max_cols() / 4)))
                vmat_params['step_size_f'] = intial_step_size
                vmat_params['step_size_b'] = intial_step_size
                best_obj_value = sol['actual_obj_value']
                self.arcs.update_best_solution()
                inner_iteration = inner_iteration + 1
                sol['accept'] = True
                sol['inner_iteration'] = inner_iteration

            else:
                if sol['actual_obj_value'] < best_obj_value:
                    sol['accept'] = True
                    print('solution accepted')
                    sol['inner_iteration'] = inner_iteration
                    self.arcs.update_best_solution()
                    self.best_iteration = self.outer_iteration
                    sol = self.arcs.calculate_dose(inf_matrix=self.inf_matrix, sol=sol, vmat_params=vmat_params, best_plan=True)
                    inner_iteration = inner_iteration + 1

                    relative_error = (best_obj_value - sol['actual_obj_value']) / best_obj_value * 100
                    if (self.outer_iteration > vmat_params['min_iteration_threshold'] and vmat_params['step_size_f'] == 1
                            and relative_error < vmat_params['termination_gap']):
                        self.outer_iteration = self.outer_iteration + 1
                        break
                    best_obj_value = sol['actual_obj_value']  # update best objective value

                    # change forward backward
                    vmat_params['forward_backward'] = (vmat_params['forward_backward'] + 1) % 2
                    self.arcs.update_leaf_pos(forward_backward=vmat_params['forward_backward'])
                    vmat_params['step_size_f'] = vmat_params['step_size_f'] + vmat_params['step_size_increment']
                    vmat_params['step_size_b'] = vmat_params['step_size_b'] + vmat_params['step_size_increment']

                else:
                    sol['accept'] = False
                    print('solution rejected..')
                    sol['inner_iteration'] = inner_iteration
                    if vmat_params['step_size_f'] > 1:
                        vmat_params['step_size_f'] = int(np.ceil(vmat_params['step_size_f'] / 2))
                        vmat_params['step_size_b'] = int(np.ceil(vmat_params['step_size_b'] / 2))
                    else:
                        if (not sol_convergence[self.outer_iteration - 1]['accept']) and (sol_convergence[self.outer_iteration - 1]['forward_backward'] == ((vmat_params['forward_backward'] + 1) % 2)) and \
                                vmat_params['step_size_f'] == 1:
                            sol['accept'] = True
                            break
                        else:
                            vmat_params['forward_backward'] = (vmat_params['forward_backward'] + 1) % 2
                            self.arcs.update_leaf_pos(forward_backward=vmat_params['forward_backward'], update_reference_leaf_pos=False)

            self.outer_iteration = self.outer_iteration + 1
        sol = sol_convergence[self.best_iteration]
        sol['inf_matrix'] = self.inf_matrix # point to influence matrix object
        return sol, sol_convergence

    def create_cvxpy_intermediate_problem_prediction(self, pred_dose_1d, final_dose_1d=None, opt_dose_1d=None):
        """

        Creates intermediate cvxpy problem for optimizing interior and boundary beamlets
        :return: None

        """
        # unpack data
        my_plan = self.my_plan
        inf_matrix = self.inf_matrix
        clinical_criteria = self.clinical_criteria
        self.obj = []
        self.constraints = []
        obj = self.obj
        constraints = self.constraints
        x = self.vars['x']
        m = inf_matrix.A.shape[0]
        if self.outer_iteration == 0:
            self.create_cvx_params()
        inf_int, inf_bound_l, inf_bound_r = self.create_interior_and_boundary_inf_matrix()

        # get interior and boundary beamlets properties in matrix form
        map_int_v = self.cvxpy_params['map_int_v']
        min_bound_index_l = self.cvxpy_params['min_bound_index_l']
        not_empty_bound_l = self.cvxpy_params['not_empty_bound_l']
        current_leaf_pos_l = self.cvxpy_params['current_leaf_pos_l']
        card_bound_inds_l = self.cvxpy_params['card_bound_inds_l']
        min_bound_index_r = self.cvxpy_params['min_bound_index_r']
        not_empty_bound_r = self.cvxpy_params['not_empty_bound_r']
        current_leaf_pos_r = self.cvxpy_params['current_leaf_pos_r']
        card_bound_inds_r = self.cvxpy_params['card_bound_inds_r']
        map_adj_int = self.cvxpy_params['map_adj_int']
        map_adj_bound = self.cvxpy_params['map_adj_bound']
        total_rows = np.sum([arc['total_rows'] for arc in self.arcs.arcs_dict['arcs']])
        total_beams = np.sum([arc['num_beams'] for arc in self.arcs.arcs_dict['arcs']])
        num_fractions = clinical_criteria.get_num_of_fractions()

        # Construct optimization problem
        # create variables
        leaf_pos_mu_l = cp.Variable(total_rows, pos=True)
        leaf_pos_mu_r = cp.Variable(total_rows, pos=True)
        int_v = cp.Variable(total_beams, pos=True)
        bound_v_l = cp.Variable(total_rows, pos=True)
        bound_v_r = cp.Variable(total_rows, pos=True)

        # save required variables in optimization object for future use
        self.vars['leaf_pos_mu_l'] = leaf_pos_mu_l
        self.vars['leaf_pos_mu_r'] = leaf_pos_mu_r
        self.vars['int_v'] = int_v
        self.vars['bound_v_l'] = bound_v_l
        self.vars['bound_v_r'] = bound_v_r
        ptv_vox = inf_matrix.get_opt_voxels_idx('PTV')
        if final_dose_1d is None:
            final_dose_1d = np.zeros(inf_matrix.A.shape[0])
        if opt_dose_1d is None:
            opt_dose_1d = np.zeros(inf_matrix.A.shape[0])
        # voxel weights for oar objectives
        all_vox = np.arange(m)
        oar_voxels = all_vox[~np.isin(np.arange(m), ptv_vox)]
        obj += [
            100*(1 / len(ptv_vox)) * cp.sum_squares((inf_int[ptv_vox, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[ptv_vox, :] @ cp.multiply(bound_v_l, map_adj_bound)
                                                     + inf_bound_r[ptv_vox, :] @ cp.multiply(bound_v_r, map_adj_bound) + final_dose_1d[ptv_vox] - opt_dose_1d[ptv_vox]) - (pred_dose_1d[ptv_vox] / num_fractions))]
        obj += [
            0.1 * (1 / len(ptv_vox)) * cp.sum_squares((inf_int[ptv_vox, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[ptv_vox, :] @ cp.multiply(bound_v_l, map_adj_bound)
                                                       + inf_bound_r[ptv_vox, :] @ cp.multiply(bound_v_r, map_adj_bound) + final_dose_1d[ptv_vox] - opt_dose_1d[ptv_vox]) - (my_plan.get_prescription() / my_plan.get_num_of_fractions()))]

        dO = cp.Variable(oar_voxels.shape[0], pos=True)
        constraints += [(inf_int[oar_voxels, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[oar_voxels, :] @ cp.multiply(bound_v_l, map_adj_bound)
                         + inf_bound_r[oar_voxels, :] @ cp.multiply(bound_v_r, map_adj_bound) + final_dose_1d[oar_voxels] - opt_dose_1d[oar_voxels]) <= pred_dose_1d[oar_voxels] / num_fractions + dO]
        obj += [(1 / dO.shape[0]) * cp.sum_squares(dO)]
        obj += [0.0001 * (1 / dO.shape[0]) * cp.sum_squares(inf_int[oar_voxels, :] @ cp.multiply(int_v, map_adj_int) + inf_bound_l[oar_voxels, :] @ cp.multiply(bound_v_l, map_adj_bound)
                                                            + inf_bound_r[oar_voxels, :] @ cp.multiply(bound_v_r, map_adj_bound) + final_dose_1d[oar_voxels] - opt_dose_1d[oar_voxels])]

        apt_reg_m = self.cvxpy_params['apt_reg_m']
        card_ar = self.cvxpy_params['card_ar']
        weight = 1 * (my_plan.get_prescription() / my_plan.get_num_of_fractions())
        obj += [weight / card_ar * (cp.sum(cp.sum_squares(apt_reg_m @ leaf_pos_mu_l)) + cp.sum(
            cp.sum_squares(apt_reg_m @ leaf_pos_mu_r)))]

        apt_sim_m = self.cvxpy_params['apt_sim_m']
        card_as = self.cvxpy_params['card_as']
        weight = 1 * (my_plan.get_prescription() / my_plan.get_num_of_fractions())
        obj += [weight / card_as * (cp.sum(cp.sum_squares(apt_sim_m @ leaf_pos_mu_l)) + cp.sum(
            cp.sum_squares(apt_sim_m @ leaf_pos_mu_r)))]

        similar_mu_obj = []
        index_stop = 0
        index_start = 0
        print('Objective for similar MU between consecutive control points added..')
        for arc in self.arcs.arcs_dict['arcs']:
            index_stop += arc['num_beams']
            for j in range(index_start, index_stop - 1):
                similar_mu_obj += [1 * cp.abs(int_v[j] - int_v[j + 1])]
            index_start += arc['num_beams']
        obj += [cp.sum(similar_mu_obj)]

        # Create convex leaf positions
        constraints += [
            leaf_pos_mu_l == cp.multiply(cp.multiply(1 - not_empty_bound_l, current_leaf_pos_l), int_v[map_int_v]) +
            cp.multiply(cp.multiply(not_empty_bound_l, min_bound_index_l), int_v[map_int_v])
            + cp.multiply((int_v[map_int_v] - bound_v_l), card_bound_inds_l)]
        constraints += [
            leaf_pos_mu_r == cp.multiply(cp.multiply(1 - not_empty_bound_r, current_leaf_pos_r), int_v[map_int_v]) +
            cp.multiply(cp.multiply(not_empty_bound_r, min_bound_index_r), int_v[map_int_v])
            + cp.multiply(bound_v_r, card_bound_inds_r)]
        # generic constraints for relation between interior and boundary beamlets
        constraints += [leaf_pos_mu_r - leaf_pos_mu_l >= int_v[map_int_v]]
        constraints += [int_v >= self.vmat_params['mu_min']]
        constraints += [bound_v_l <= int_v[map_int_v]]
        constraints += [bound_v_r <= int_v[map_int_v]]
        if 'minimum_dynamic_leaf_gap_mm' in self.vmat_params:
            min_leaf_gap_beamlet = self.vmat_params['minimum_dynamic_leaf_gap_mm'] / my_plan.beams.get_beamlet_width() * 1.01
            constraints += [leaf_pos_mu_r - leaf_pos_mu_l >= int_v[map_int_v] * min_leaf_gap_beamlet]

    def calc_actual_objective_value_prediction(self, sol: dict, pred_dose_1d):
        """
        Calculate actual objective function value using actual solution

        """
        # unpack data and optimization problems
        inf_matrix = self.my_plan.inf_matrix
        num_fractions = self.my_plan.get_num_of_fractions()
        ptv_vox = inf_matrix.get_opt_voxels_idx('PTV')
        # voxel weights for oar objectives
        m = inf_matrix.A.shape[0]
        all_vox = np.arange(m)
        oar_voxels = all_vox[~np.isin(np.arange(m), ptv_vox)]

        ptv_obj = 100*(1 / len(ptv_vox)) * np.sum((sol['act_dose_v'][ptv_vox] - (pred_dose_1d[ptv_vox] / num_fractions)) ** 2)
        ptv_obj1 = 0.1 * (1 / len(ptv_vox)) * np.sum((sol['act_dose_v'][ptv_vox] - (self.my_plan.get_prescription() / num_fractions)) ** 2)
        oar_obj = (1 / len(oar_voxels)) * np.sum(np.maximum(sol['act_dose_v'][oar_voxels] - (pred_dose_1d[oar_voxels] / num_fractions), 0)** 2)
        oar_obj1 = 0.0001*(1 / len(oar_voxels)) * np.sum(np.maximum(sol['act_dose_v'][oar_voxels] - (pred_dose_1d[oar_voxels] / num_fractions), 0) ** 2)
        apt_reg_obj = self.obj[4].value
        apt_sim_obj = self.obj[5].value
        similar_mu_obj = self.obj[6].value
        sol['actual_obj_value'] = np.round((ptv_obj + ptv_obj1 + oar_obj1 + oar_obj + apt_reg_obj + apt_sim_obj + similar_mu_obj), 4) #+ apt_reg_obj + apt_sim_obj + similar_mu_obj), 4)
        return sol

    def run_sequential_cvx_algo_prediction(self, pred_dose_1d, *args, **kwargs):
        """
        :param pred_dose_1d: predicted dose 1d array
        Returns sol and convergence of the sequential convex algorithm for optimizing the plan.
        Solver parameters can be passed in args.

        """
        # running scp algorithm:
        inner_iteration = int(0)
        best_obj_value = 0
        vmat_params = self.vmat_params
        self.arcs.get_initial_leaf_pos(initial_leaf_pos=vmat_params['initial_leaf_pos'])
        sol_convergence = []
        while True:

            self.arcs.gen_interior_and_boundary_beamlets(forward_backward=vmat_params['forward_backward'], step_size_f=vmat_params['step_size_f'], step_size_b=vmat_params['step_size_b'])
            # Optimize using the predicted plan
            self.create_cvxpy_intermediate_problem_prediction(pred_dose_1d=pred_dose_1d)
            sol = self.solve(*args, **kwargs)
            sol_convergence.append(sol)

            # post processing
            self.arcs.calc_actual_from_intermediate_sol(sol)
            sol = self.arcs.calculate_dose(inf_matrix=self.inf_matrix, sol=sol, vmat_params=vmat_params, best_plan=False)
            sol = self.calc_actual_objective_value_prediction(sol, pred_dose_1d=pred_dose_1d)

            if inner_iteration == 0:

                intial_step_size = int(np.maximum(3, np.ceil(self.arcs.get_max_cols() / 4)))
                vmat_params['step_size_f'] = intial_step_size
                vmat_params['step_size_b'] = intial_step_size
                best_obj_value = sol['actual_obj_value']
                self.arcs.update_best_solution()
                sol['inner_iteration'] = inner_iteration
                inner_iteration = inner_iteration + 1
                sol['accept'] = True

            else:
                if sol['actual_obj_value'] < best_obj_value:
                    sol['accept'] = True
                    print('solution accepted')
                    sol['inner_iteration'] = inner_iteration
                    self.arcs.update_best_solution()
                    self.best_iteration = self.outer_iteration
                    sol = self.arcs.calculate_dose(inf_matrix=self.inf_matrix, sol=sol, vmat_params=vmat_params, best_plan=True)
                    inner_iteration = inner_iteration + 1

                    relative_error = (best_obj_value - sol['actual_obj_value']) / best_obj_value * 100
                    if (self.outer_iteration > vmat_params['min_iteration_threshold'] and vmat_params['step_size_f'] == 1
                            and relative_error < vmat_params['termination_gap']):
                        self.outer_iteration = self.outer_iteration + 1
                        break
                    best_obj_value = sol['actual_obj_value']  # update best objective value

                    # change forward backward
                    vmat_params['forward_backward'] = (vmat_params['forward_backward'] + 1) % 2
                    self.arcs.update_leaf_pos(forward_backward=vmat_params['forward_backward'])
                    vmat_params['step_size_f'] = vmat_params['step_size_f'] + vmat_params['step_size_increment']
                    vmat_params['step_size_b'] = vmat_params['step_size_b'] + vmat_params['step_size_increment']

                else:
                    sol['accept'] = False
                    print('solution rejected..')
                    sol['inner_iteration'] = inner_iteration
                    if vmat_params['step_size_f'] > 1:
                        vmat_params['step_size_f'] = int(np.ceil(vmat_params['step_size_f'] / 2))
                        vmat_params['step_size_b'] = int(np.ceil(vmat_params['step_size_b'] / 2))
                    else:
                        if (not sol_convergence[self.outer_iteration - 1]['accept']) and (sol_convergence[self.outer_iteration - 1]['forward_backward'] == ((vmat_params['forward_backward'] + 1) % 2)) and \
                                vmat_params['step_size_f'] == 1:
                            sol['accept'] = True
                            break
                        else:
                            vmat_params['forward_backward'] = (vmat_params['forward_backward'] + 1) % 2
                            self.arcs.update_leaf_pos(forward_backward=vmat_params['forward_backward'], update_reference_leaf_pos=False)

            self.outer_iteration = self.outer_iteration + 1
        sol = sol_convergence[self.best_iteration]
        sol['inf_matrix'] = self.inf_matrix # point to influence matrix object
        return sol, sol_convergence

    def run_sequential_cvx_algo_prediction_correction(self, pred_dose_1d, final_dose_1d, opt_dose_1d, cvxpy_params, vmat_params, *args, **kwargs):
        """
        :param pred_dose_1d: predicted dose 1d array
        Returns sol and convergence of the sequential convex algorithm for optimizing the plan.
        Solver parameters can be passed in args.

        """
        # running scp algorithm:
        inner_iteration = int(0)
        best_obj_value = 0
        self.vmat_params = vmat_params
        self.vmat_params['step_size_f'] = 1
        self.vmat_params['step_size_b'] = 1
        self.vmat_params['initial_step_size'] = 1
        self.vmat_params['step_size_increment'] = 0
        self.vmat_params['termination_gap'] = 0.5
        self.cvxpy_params = cvxpy_params

        # self.arcs.get_initial_leaf_pos(initial_leaf_pos=vmat_params['initial_leaf_pos'])
        sol_convergence = []
        self.outer_iteration = 1
        while True:
            if self.outer_iteration > 1:
                self.arcs.gen_interior_and_boundary_beamlets(forward_backward=vmat_params['forward_backward'], step_size_f=vmat_params['step_size_f'], step_size_b=vmat_params['step_size_b'])
            # Optimize using the predicted plan
            self.create_cvxpy_intermediate_problem_prediction(pred_dose_1d=pred_dose_1d, final_dose_1d=final_dose_1d, opt_dose_1d=opt_dose_1d)
            sol = self.solve(*args, **kwargs)
            sol_convergence.append(sol)

            # post processing
            self.arcs.calc_actual_from_intermediate_sol(sol)
            sol = self.arcs.calculate_dose(inf_matrix=self.inf_matrix, sol=sol, vmat_params=vmat_params, best_plan=False)
            sol['act_dose_v'] = sol['act_dose_v'] + final_dose_1d - opt_dose_1d
            sol['int_dose_v'] = sol['int_dose_v'] + final_dose_1d - opt_dose_1d
            sol = self.calc_actual_objective_value_prediction(sol, pred_dose_1d=pred_dose_1d)

            if inner_iteration == 0:

                self.arcs.update_leaf_pos(forward_backward=vmat_params['forward_backward'])
                best_obj_value = sol['actual_obj_value']
                self.arcs.update_best_solution()
                sol['inner_iteration'] = inner_iteration
                inner_iteration = inner_iteration + 1
                sol['accept'] = True

            else:
                if sol['actual_obj_value'] < best_obj_value:
                    sol['accept'] = True
                    print('solution accepted')
                    sol['inner_iteration'] = inner_iteration
                    self.arcs.update_best_solution()
                    self.best_iteration = self.outer_iteration
                    sol = self.arcs.calculate_dose(inf_matrix=self.inf_matrix, sol=sol, vmat_params=vmat_params, best_plan=True)
                    sol['best_act_dose_v'] = sol['best_act_dose_v'] + final_dose_1d - opt_dose_1d
                    inner_iteration = inner_iteration + 1

                    relative_error = (best_obj_value - sol['actual_obj_value']) / best_obj_value * 100
                    if self.outer_iteration > 15:
                        self.outer_iteration = self.outer_iteration + 1
                        break
                    if vmat_params['step_size_f'] == 1 and relative_error < vmat_params['termination_gap']:
                        self.outer_iteration = self.outer_iteration + 1
                        break
                    best_obj_value = sol['actual_obj_value']  # update best objective value

                    # change forward backward
                    vmat_params['forward_backward'] = (vmat_params['forward_backward'] + 1) % 2
                    self.arcs.update_leaf_pos(forward_backward=vmat_params['forward_backward'])
                    vmat_params['step_size_f'] = vmat_params['step_size_f'] + vmat_params['step_size_increment']
                    vmat_params['step_size_b'] = vmat_params['step_size_b'] + vmat_params['step_size_increment']

                else:
                    sol['accept'] = False
                    print('solution rejected..')
                    sol['inner_iteration'] = inner_iteration
                    if vmat_params['step_size_f'] > 1:
                        vmat_params['step_size_f'] = int(np.ceil(vmat_params['step_size_f'] / 2))
                        vmat_params['step_size_b'] = int(np.ceil(vmat_params['step_size_b'] / 2))
                    else:
                        if (not sol_convergence[-2]['accept']) and (sol_convergence[-2]['forward_backward'] == ((vmat_params['forward_backward'] + 1) % 2)) and \
                                vmat_params['step_size_f'] == 1:
                            sol['accept'] = True
                            break
                        else:
                            vmat_params['forward_backward'] = (vmat_params['forward_backward'] + 1) % 2
                            self.arcs.update_leaf_pos(forward_backward=vmat_params['forward_backward'], update_reference_leaf_pos=False)

            self.outer_iteration = self.outer_iteration + 1
        # sol['inf_matrix'] = self.inf_matrix # point to influence matrix object
        return sol, sol_convergence
    def solve(self, actual_sol_correction=False, return_cvxpy_prob=False, sol: dict = None, *args, **kwargs):
        """
                Return optimal solution and influence matrix associated with it in the form of dictionary
                If return_problem set to true, returns cvxpy problem instance

                :Example
                        dict = {"act_dose_v": [..],
                        "int_v":[..],
                        "bound_v_l":[..],
                        "bound_v_r":[..],
                        "inf_matrix": my_plan.inf_marix
                        }

                :return: solution dictionary, cvxpy problem instance(optional)
                """

        if actual_sol_correction:
            problem = cp.Problem(cp.Minimize(cp.sum(self.obj_actual)), constraints=self.constraints_actual)
            print("#####################################################################\n")
            print("solving actual problem for outer iteration:{}, step size:{}".format(self.outer_iteration, self.vmat_params['step_size_f']))

        else:
            problem = cp.Problem(cp.Minimize(cp.sum(self.obj)), constraints=self.constraints)
            print("#####################################################################\n")
            print("solving intermediate problem for outer iteration:{}, step size:{}".format(self.outer_iteration, self.vmat_params['step_size_f']))
        print('Running Optimization..')
        t = time.time()
        problem.solve(*args, **kwargs)
        elapsed = time.time() - t
        print("Optimal value: %s" % problem.value)
        if problem.solver_stats.setup_time is not None:
            print("Setup time for solver: {} seconds".format(problem.solver_stats.setup_time))
        print("Solve time: {} seconds".format(problem.solver_stats.solve_time))
        print("Elapsed time: {} seconds".format(elapsed))

        if not actual_sol_correction:
            sol = dict()
            sol['outer_iteration'] = self.outer_iteration
            sol['step_size_f_b'] = self.vmat_params['forward_backward'] * self.vmat_params['step_size_f'] + (
                        1 - self.vmat_params['forward_backward']) * self.vmat_params['step_size_b']
            sol['forward_backward'] = self.vmat_params['forward_backward']
            sol['intermediate_obj_value'] = np.round(problem.value, 4)
            for key, value in self.vars.items():
                if key in ['leaf_pos_mu_l', 'leaf_pos_mu_r', 'int_v', 'bound_v_l', 'bound_v_r']:
                    sol[key] = np.round(value.value, 6)
            sol['solver_stats'] = deepcopy(problem.solver_stats)
        else:
            sol['beam_mu'] = np.round(self.vars['beam_mu'].value, 6)
        sol['time_seconds'] = np.round(elapsed)
        sol['solver_stats'] = deepcopy(problem.solver_stats)
        if return_cvxpy_prob:
            return sol, problem
        else:
            return sol

    def dose_to_gy(self, key, value):
        if "prescription_gy" in str(value):
            prescription_gy = self.clinical_criteria.get_prescription()
            return eval(value)
        elif 'gy' in key:
            return value
        elif 'perc' in key:
            return value*self.clinical_criteria.get_prescription()/100