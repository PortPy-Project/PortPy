from __future__ import annotations
import numpy as np
import cvxpy as cp
import time
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from portpy_photon.plan import Plan
    from portpy_photon.influence_matrix import InfluenceMatrix


class Optimization(object):
    """
    Optimization class for optimizing and creating the plan
    """

    @staticmethod
    def run_IMRT_fluence_map_CVXPy(my_plan: Plan, inf_matrix: InfluenceMatrix = None, solver: str = 'MOSEK',
                                   verbose: bool = True, cvxpy_options: dict = None, **opt_params) -> dict:
        """
        It runs optimization to create optimal plan based upon clinical criteria

        :param my_plan: object of class Plan
        :param inf_matrix: object of class InfluenceMatrix
        :param solver: default solver 'MOSEK'. check cvxpy website for available solvers
        :param verbose: Default to True. If set to False, it will not print the solver iterations.
        :param cvxpy_options: cvxpy and the solver settings
        :param opt_params: optimization parameters for modifying parameters of problem statement
        :return: returns the solution dictionary in format of:
            dict: {
                   'optimal_intensity': list(float),
                   'dose_1d': list(float),
                   'inf_matrix': Pointer to object of InfluenceMatrix }
                  }
        :Example:
        >>> Optimization.run_IMRT_fluence_map_CVXPy(my_plan=my_plan,inf_matrix=inf_matrix,solver='MOSEK')
        """

        if cvxpy_options is None:
            cvxpy_options = dict()

        t = time.time()

        # get data for optimization
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        A = inf_matrix.A
        cc_dict = my_plan.clinical_criteria.clinical_criteria_dict
        criteria = cc_dict['criteria']
        pres = cc_dict['pres_per_fraction_gy']
        num_fractions = cc_dict['num_of_fractions']
        [Qx, Qy, num_rows, num_cols] = Optimization.get_smoothness_matrix(inf_matrix.beamlets_dict)
        st = inf_matrix

        # create and add rind constraints
        rinds = ['RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
        if rinds[0] not in my_plan.structures.structures_dict['name']:  # check if rind is already created. If yes, skip rind creation
            Optimization.create_rinds(my_plan, size_mm=[5, 5, 20, 30, 500])
            Optimization.set_rinds_opt_voxel_idx(my_plan,
                                                 inf_matrix=inf_matrix)  # rind_0 is 5mm after PTV, rind_2 is 5 mm after rind_1, and so on..
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
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('CORD')))] = 10
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('ESOPHAGUS')))] = 20
            if 'HEART' in st.opt_voxels_dict['name']:
                oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('HEART')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3

        # Construct the problem.
        x = cp.Variable(A.shape[1], pos=True)
        dO = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)

        # Form objective.
        ptv_overdose_weight = opt_params['ptv_overdose_weight'] if 'ptv_overdose_weight' in opt_params else 10000
        ptv_underdose_weight = opt_params['ptv_underdose_weight'] if 'ptv_underdose_weight' in opt_params else 100000
        smoothness_weight = opt_params['smoothness_weight'] if 'smoothness_weight' in opt_params else 10
        total_oar_weight = opt_params['total_oar_weight'] if 'total_oar_weight' in opt_params else 10
        smoothness_X_weight = 0.6
        smoothness_Y_weight = 0.4

        print('Objective Start')
        obj = [(1 / len(st.get_opt_voxels_idx('PTV'))) * (
                ptv_overdose_weight * cp.sum_squares(dO) + ptv_underdose_weight * cp.sum_squares(dU)),
               smoothness_weight * (
                       smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) + smoothness_Y_weight * (
                           1 / num_rows) * cp.sum_squares(Qy @ x)),
               total_oar_weight * (1 / A[oar_voxels, :].shape[0]) * cp.sum_squares(
                   cp.multiply(cp.sqrt(oar_weights), A[oar_voxels, :] @ x))]

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

        # Step 1 and 2 constraints
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x <= pres + dO]
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x >= pres - dU]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)

        print('Problem loaded')
        prob.solve(solver=solver, verbose=verbose, **cvxpy_options)
        print("optimal value with {}:{}".format(solver, prob.value))
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))

        # saving optimal solution to the solution dictionary
        sol = {'optimal_intensity': x.value.astype('float32'), 'inf_matrix': inf_matrix}

        return sol

    @staticmethod
    def run_IMRT_fluence_map_CVXPy_dvh_benchmark(my_plan: Plan, inf_matrix: InfluenceMatrix = None,
                                                 dvh_criteria: List[dict] = None, solver: str = 'MOSEK',
                                                 verbose: bool = True, cvxpy_options: dict = None,
                                                 **opt_params) -> dict:
        """
        Add dvh constraints and solve the optimization for getting ground truth solution

        :param inf_matrix: object of class InfluenceMatrix
        :param my_plan: object of class Plan
        :param solver: default solver 'MOSEK'. check cvxpy website for available solvers
        :param dvh_criteria: dvh criteria to be met in optimization. Defaults to None.
        If None, it will optimize for all the dvh criteria in clinical criteria.
        :param verbose: Default to True. If set to False, it will not print the solver iterations.
        :param cvxpy_options: cvxpy and the solver settings
        :return: save optimal solution to plan object called opt_sol


        """
        if cvxpy_options is None:
            cvxpy_options = dict()
        t = time.time()

        # get data for optimization
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        A = inf_matrix.A
        cc_dict = my_plan.clinical_criteria.clinical_criteria_dict
        criteria = cc_dict['criteria']
        pres = cc_dict['pres_per_fraction_gy']
        num_fractions = cc_dict['num_of_fractions']
        [Qx, Qy, num_rows, num_cols] = Optimization.get_smoothness_matrix(inf_matrix.beamlets_dict)
        st = inf_matrix

        # create and add rind constraints
        rinds = ['RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
        if rinds[0] not in my_plan.structures.structures_dict[
            'name']:  # check if rind is already created. If yes, skip rind creation
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
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('CORD')))] = 10
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('ESOPHAGUS')))] = 20
            if 'HEART' in st.opt_voxels_dict['name']:
                oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('HEART')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3

        # get volume constraints in V(Gy) <= v% format
        import pandas as pd
        df_dvh_criteria = pd.DataFrame()
        count = 0
        if dvh_criteria is None:
            dvh_criteria = criteria

        for i in range(len(dvh_criteria)):
            if 'dose_volume' in dvh_criteria[i]['name']:
                limit_key = Optimization.matching_keys(dvh_criteria[i]['constraints'], 'limit')
                if limit_key in dvh_criteria[i]['constraints']:
                    df_dvh_criteria.at[count, 'structure'] = dvh_criteria[i]['parameters']['structure_name']
                    df_dvh_criteria.at[count, 'dose_gy'] = dvh_criteria[i]['parameters']['dose_gy']

                    # getting max dose_1d for the same structure
                    max_dose_struct = 1000
                    for j in range(len(criteria)):
                        if 'max_dose' in criteria[j]['name']:
                            if 'limit_dose_gy' in criteria[j]['constraints']:
                                org = criteria[j]['parameters']['structure_name']
                                if org == dvh_criteria[i]['parameters']['structure_name']:
                                    max_dose_struct = criteria[j]['constraints']['limit_dose_gy']
                    df_dvh_criteria.at[count, 'M'] = max_dose_struct - dvh_criteria[i]['parameters']['dose_gy']
                    if 'perc' in limit_key:
                        df_dvh_criteria.at[count, 'vol_perc'] = dvh_criteria[i]['constraints'][limit_key]
                    count = count + 1

        # Construct the problem.
        x = cp.Variable(A.shape[1], pos=True)
        dO = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)

        # binary variable for dvh constraints
        b = cp.Variable(
            len(np.concatenate([st.get_opt_voxels_idx(org) for org in df_dvh_criteria.structure.to_list()])),
            boolean=True)
        # Form objective.
        ptv_overdose_weight = opt_params['ptv_overdose_weight'] if 'ptv_overdose_weight' in opt_params else 10000
        ptv_underdose_weight = opt_params['ptv_underdose_weight'] if 'ptv_underdose_weight' in opt_params else 100000
        smoothness_weight = opt_params['smoothness_weight'] if 'smoothness_weight' in opt_params else 10
        total_oar_weight = opt_params['total_oar_weight'] if 'total_oar_weight' in opt_params else 10
        smoothness_X_weight = 0.6
        smoothness_Y_weight = 0.4

        print('Objective Start')
        obj = [(1 / len(st.get_opt_voxels_idx('PTV'))) * (
                ptv_overdose_weight * cp.sum_squares(dO) + ptv_underdose_weight * cp.sum_squares(dU)),
               smoothness_weight * (
                       smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) + smoothness_Y_weight * (
                           1 / num_rows) * cp.sum_squares(Qy @ x)),
               total_oar_weight * (1 / A[oar_voxels, :].shape[0]) * cp.sum_squares(
                   cp.multiply(cp.sqrt(oar_weights), A[oar_voxels, :] @ x))]

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
        start = 0
        for i in range(len(df_dvh_criteria)):
            struct, limit, v, M = df_dvh_criteria.loc[i, 'structure'], df_dvh_criteria.loc[i, 'dose_gy'], \
                                  df_dvh_criteria.loc[i, 'vol_perc'], df_dvh_criteria.loc[i, 'M']
            end = start + len(st.get_opt_voxels_idx(struct))
            frac = my_plan.structures.get_fraction_of_vol_in_calc_box(struct)
            constraints += [A[st.get_opt_voxels_idx(struct), :] @ x <= limit / num_fractions + b[start:end] * M / num_fractions]
            constraints += [b @ st.get_opt_voxels_size(struct) <= (v / frac) / 100 * sum(
                st.get_opt_voxels_size(struct))]
            start = end

        # Step 1 and 2 constraints
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x <= pres + dO]
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x >= pres - dU]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)
        # Defining the constraints
        print('Problem loaded')
        prob.solve(solver=solver, verbose=verbose, **cvxpy_options)
        print("optimal value with {}:{}".format(solver, prob.value))
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))

        # saving optimal solution to my_plan
        sol = {'optimal_intensity': x.value, 'inf_matrix': inf_matrix}

        return sol

    @staticmethod
    def run_IMRT_fluence_map_CVXPy_BOO_benchmark(my_plan, inf_matrix=None, num_beams: int = 7, solver: str = 'MOSEK',
                                                 verbose: bool = True, cvxpy_options: dict = None, **opt_params):
        """
        Creates MIP problem for selecting optimal beam angles

        :param inf_matrix: object of class InfluenceMatrix
        :param my_plan: object of class Plan
        :param solver: default solver 'MOSEK'. check cvxpy website for available solvers.
        :param num_beams: Default to 7. Numbers of beams to be selected from the pool of beams
        :param verbose: Default to True. If set to False, it will not print the solver iterations.
        :param cvxpy_options: cvxpy and the solver settings
        :return: save optimal solution to plan object called opt_sol


        """
        if cvxpy_options is None:
            cvxpy_options = dict()
        t = time.time()

        # get data for optimization
        if inf_matrix is None:
            inf_matrix = my_plan.inf_matrix
        A = inf_matrix.A
        cc_dict = my_plan.clinical_criteria.clinical_criteria_dict
        criteria = cc_dict['criteria']
        pres = cc_dict['pres_per_fraction_gy']
        num_fractions = cc_dict['num_of_fractions']
        [Qx, Qy, num_rows, num_cols] = Optimization.get_smoothness_matrix(inf_matrix.beamlets_dict)
        st = inf_matrix

        # create and add rind constraints
        rinds = ['RIND_0', 'RIND_1', 'RIND_2', 'RIND_3', 'RIND_4']
        if rinds[0] not in my_plan.structures.structures_dict[
            'name']:  # check if rind is already created. If yes, skip rind creation
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
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('CORD')))] = 10
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('ESOPHAGUS')))] = 20
            if 'HEART' in st.opt_voxels_dict['name']:
                oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('HEART')))] = 20
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_0')))] = 3
            oar_weights[np.where(np.isin(oar_voxels, st.get_opt_voxels_idx('RIND_1')))] = 3

        # Construct the problem.
        x = cp.Variable(A.shape[1], pos=True)
        dO = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)
        dU = cp.Variable(len(st.get_opt_voxels_idx('PTV')), pos=True)

        # binary variable for selecting beams
        b = cp.Variable(len(inf_matrix.beamlets_dict), boolean=True)

        # Form objective.
        ptv_overdose_weight = opt_params['ptv_overdose_weight'] if 'ptv_overdose_weight' in opt_params else 10000
        ptv_underdose_weight = opt_params['ptv_underdose_weight'] if 'ptv_underdose_weight' in opt_params else 100000
        smoothness_weight = opt_params['smoothness_weight'] if 'smoothness_weight' in opt_params else 10
        total_oar_weight = opt_params['total_oar_weight'] if 'total_oar_weight' in opt_params else 10
        smoothness_X_weight = 0.6
        smoothness_Y_weight = 0.4

        print('Objective Start')
        obj = [(1 / len(st.get_opt_voxels_idx('PTV'))) * (
                ptv_overdose_weight * cp.sum_squares(dO) + ptv_underdose_weight * cp.sum_squares(dU)),
               smoothness_weight * (
                       smoothness_X_weight * (1 / num_cols) * cp.sum_squares(Qx @ x) + smoothness_Y_weight * (
                           1 / num_rows) * cp.sum_squares(Qy @ x)),
               total_oar_weight * (1 / A[oar_voxels, :].shape[0]) * cp.sum_squares(
                   cp.multiply(cp.sqrt(oar_weights), A[oar_voxels, :] @ x))]

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

        #  Constraints for selecting beams
        constraints += [cp.sum(b) <= num_beams]
        for i in range(len(inf_matrix.beamlets_dict)):
            start_beamlet = inf_matrix.beamlets_dict[i]['start_beamlet']
            end_beamlet = inf_matrix.beamlets_dict[i]['end_beamlet']
            M = 40  # upper bound on the beamlet intensity
            constraints += [x[start_beamlet:end_beamlet] <= b[i] * M]

        # Step 1 and 2 constraints
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x <= pres + dO]
        constraints += [A[st.get_opt_voxels_idx('PTV'), :] @ x >= pres - dU]

        print('Constraints Done')

        prob = cp.Problem(cp.Minimize(sum(obj)), constraints)
        # Defining the constraints
        print('Problem loaded')
        prob.solve(solver=solver, verbose=verbose, **cvxpy_options)
        print("optimal value with {}:{}".format(solver, prob.value))
        elapsed = time.time() - t
        print('Elapsed time {} seconds'.format(elapsed))

        # saving optimal solution to my_plan
        sol = {'optimal_intensity': x.value, 'optimal_beams': b.value, 'inf_matrix': inf_matrix}

        return sol

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
        sRow = np.zeros((beamReq[-1]['end_beamlet'] + 1, beamReq[-1]['end_beamlet'] + 1), dtype=int)
        sCol = np.zeros((beamReq[-1]['end_beamlet'] + 1, beamReq[-1]['end_beamlet'] + 1), dtype=int)
        num_rows = 0
        num_cols = 0
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
                prev_dummy_name = 'dummy_{}'.format(ind - 1)
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
