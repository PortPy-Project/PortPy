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
from typing import List
import pandas as pd
from .data_explorer import DataExplorer
import json
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
from copy import deepcopy
import numpy as np


class ClinicalCriteria:
    """
    A class representing clinical criteria.

    - **Attributes** ::

        :param clinical_criteria_dict: dictionary containing metadata about clinical criteria

        :Example
                dict = {"disease_site": "Lung",
                "protocol_name": "Lung_Default_2Gy_30Fr",
                "pres_per_fraction_gy": 2,
                "num_of_fractions": 30,
                "criteria": [
                    {
                      "type": "max_dose",
                      "parameters": {
                        "struct": "GTV"
                      }}

    - **Methods** ::
        :add_criterion(criterion:str, parameters:dict, constraints:dict)
        :modify_criterion(criterion:str, parameters:dict, constraints:dict)

    """

    def __init__(self, data: DataExplorer = None, protocol_name: str = None, protocol_type: str = 'Default', file_name: str = None):
        """

        :param clinical_criteria: dictionary containing information about clinical criteria

        """
        if file_name is None:
            clinical_criteria_dict = data.load_config_clinical_criteria(protocol_name, protocol_type=protocol_type)
        else:
            f = open(file_name)
            clinical_criteria_dict = json.load(f)
            f.close()
        self.clinical_criteria_dict = clinical_criteria_dict
        self.dvh_table = pd.DataFrame()

    def get_prescription(self) -> float:
        """

        :return: prescription in Gy
        """
        pres = self.clinical_criteria_dict['pres_per_fraction_gy'] * \
               self.clinical_criteria_dict[
                   'num_of_fractions']
        return pres

    def get_num_of_fractions(self) -> float:
        """

        :return: number of fractions to be delivered
        """
        return self.clinical_criteria_dict['num_of_fractions']

    def add_criterion(self, type: str, parameters: dict, constraints: dict) -> None:
        """
        Add criterion to the clinical criteria dictionary

        :param type: criterion name. e.g. max_dose
        :param parameters: parameters dictionary e.g. parameters = {'struct':'PTV'}
        :param constraints: constraints dictionary e.g. constraints = {'limit_dose_gy':66, 'goal_dose_gy':60}
        :return: add the criteria to clinical criteria dictionary

        """

        self.clinical_criteria_dict['criteria'].append({'type': type})
        self.clinical_criteria_dict['criteria'][-1]['parameters'] = parameters
        self.clinical_criteria_dict['criteria'][-1]['constraints'] = constraints

    @staticmethod
    def create_criterion(type: str, parameters: dict, constraints: dict):
        """
        Create criterion and return list

        :param type: criterion name. e.g. max_dose
        :param parameters: parameters dictionary e.g. parameters = {'struct':'PTV'}
        :param constraints: constraints dictionary e.g. constraints = {'limit_dose_gy':66, 'goal_dose_gy':60}
        :return: add the criteria to clinical criteria dictionary

        """

        criterion = [{'type': type, 'parameters': parameters, 'constraints': constraints}]
        return criterion

    def get_criteria(self, type: str = None) -> List[dict]:
        """
        Returns all the clinical criteria
        :return:
        """
        all_criteria = []
        if type is None:
            all_criteria = self.clinical_criteria_dict['criteria']
        elif type == 'max_dose':
            criteria = self.clinical_criteria_dict['criteria']
            ind = [i for i in range(len(criteria)) if criteria[i]['type'] == type]
            if len(ind) > 0:
                all_criteria = [criteria[i] for i in ind]
        elif type == 'mean_dose':
            criteria = self.clinical_criteria_dict['criteria']
            ind = [i for i in range(len(criteria)) if criteria[i]['type'] == type]
            if len(ind) > 0:
                all_criteria = [criteria[i] for i in ind]

        if isinstance(all_criteria, dict):
            all_criteria = [all_criteria]
        return all_criteria

    def check_criterion_exists(self, criterion, return_ind:bool = False):
        criterion_exist = False
        criterion_ind = None
        for ind, crit in enumerate(self.clinical_criteria_dict['criteria']):
            if (crit['type'] == criterion['type']) and crit['parameters'] == criterion['parameters']:
                for constraint in crit['constraints']:
                    if constraint == criterion['constraints']:
                        criterion_exist = True
                        criterion_ind = ind
        if return_ind:
            return criterion_exist,criterion_ind
        else:
            return criterion_exist

    def modify_criterion(self, criterion):
        """
        Modify the criterion the clinical criteria


        """
        criterion_found = False
        for ind, crit in enumerate(self.clinical_criteria_dict['criteria']):
            if (crit['type'] == criterion['type']) and crit['parameters'] == criterion['parameters']:
                for constraint in crit['constraints']:
                    if constraint == criterion['constraints']:
                        self.clinical_criteria_dict['criteria'][ind]['constraints'][constraint] = criterion['constraints']
                        criterion_found = True
        if not criterion_found:
            raise Warning('No criteria  for {}'.format(criterion))


    def get_num(self, string: Union[str, float]):
        if "prescription_gy" in str(string):
            prescription_gy = self.get_prescription()
            return eval(string)
        elif isinstance(string, float) or isinstance(string, int):
            return string
        else:
            raise Exception('Invalid constraint')

    def matching_keys(self, dictionary, search_string):
        get_key = None
        for key, val in dictionary.items():
            if search_string in key:
                get_key = key
        if get_key is not None:
            return get_key
        else:
            return ''

    def dose_to_gy(self, key, value):
        if "prescription_gy" in str(value):
            prescription_gy = self.get_prescription()
            return eval(value)
        elif 'gy' in key:
            return value
        elif 'perc' in key:
            return value*self.get_prescription()/100

    @staticmethod
    def convert_dvh_to_dose_gy_vol_perc(my_plan, old_criteria):
        """
        Get dose volume in Gy and Percent format
        """
        criteria = deepcopy(old_criteria)
        struct_name = criteria['parameters']['structure_name']
        if criteria['type'] == 'dose_volume_D':
            constraint_keys = list(criteria['constraints'].keys())
            for key in constraint_keys:
                if 'perc' in key:
                    value = criteria['constraints'][key]
                    new_key = key.replace('perc', 'gy')
                    criteria['constraints'][new_key] = criteria['constraints'].pop(key)
                    criteria['constraints'][new_key] = value / 100 * my_plan.get_prescription()
            param_keys = list(criteria['parameters'].keys())
            for key in param_keys:
                if 'volume_cc' in key:
                    value = criteria['parameters'][key]
                    new_key = key.replace('cc', 'perc')
                    criteria['parameters'][new_key] = criteria['parameters'].pop(key)
                    criteria['parameters'][new_key] = value / my_plan.structures.get_volume_cc(
                        struct_name.upper()) * 100
        if criteria['type'] == 'dose_volume_V':
            constraint_keys = list(criteria['constraints'].keys())
            for key in constraint_keys:
                if 'volume_cc' in key:
                    value = criteria['constraints'][key]
                    new_key = key.replace('cc', 'perc')
                    criteria['constraints'][new_key] = criteria['constraints'].pop(key)
                    criteria['constraints'][new_key] = value / my_plan.structures.get_volume_cc(
                        struct_name.upper()) * 100
            param_keys = list(criteria['parameters'].keys())
            for key in param_keys:
                if 'dose_perc' in key:
                    value = criteria['parameters'][key]
                    new_key = key.replace('perc', 'gy')
                    criteria['parameters'][new_key] = criteria['parameters'].pop(key)
                    criteria['parameters'][new_key] = value / 100 * my_plan.get_prescription()
        return criteria


    def get_dvh_table(self, my_plan: Plan, constraint_list: list = None, opt_params: Union[list, dict] = None):
        if constraint_list is None:
            constraint_list = deepcopy(self.clinical_criteria_dict['criteria'])
        if opt_params is not None:
            # add/modify constraints definition if present in opt params
            for opt_constraint in opt_params['constraints']:
                # add constraint
                param = opt_constraint['parameters']
                if param['structure_name'] in my_plan.structures.get_structures():
                    criterion_exist, criterion_ind = self.check_criterion_exists(opt_constraint,
                                                                                              return_ind=True)
                    if criterion_exist:
                        constraint_list[criterion_ind] = opt_constraint
                    else:
                        constraint_list += [opt_constraint]

        dvh_updated_list = []
        for i, constraint in enumerate(constraint_list):
            if constraint['parameters']['structure_name'] in my_plan.structures.get_structures():
                if len(my_plan.inf_matrix.get_opt_voxels_idx(constraint['parameters']['structure_name'])) == 0:
                    continue
                updated_constraint = self.convert_dvh_to_dose_gy_vol_perc(my_plan, constraint)
                dvh_updated_list.append(updated_constraint)
        import pandas as pd
        df = pd.DataFrame()
        count = 0
        for i in range(len(dvh_updated_list)):
            if 'dose_volume_V' in dvh_updated_list[i]['type']:
                limit_key = self.matching_keys(dvh_updated_list[i]['constraints'], 'limit')
                dose_key = self.matching_keys(dvh_updated_list[i]['parameters'], 'dose_')
                if limit_key in dvh_updated_list[i]['constraints']:
                    df.at[count, 'structure_name'] = dvh_updated_list[i]['parameters']['structure_name']
                    df.at[count, 'dose_gy'] = self.dose_to_gy(dose_key, dvh_updated_list[i]['parameters'][dose_key])
                    df.at[count, 'volume_perc'] = dvh_updated_list[i]['constraints'][limit_key]
                    df.at[count, 'dvh_type'] = 'constraint'
                    count = count + 1
                goal_key = self.matching_keys(dvh_updated_list[i]['constraints'], 'goal')
                if goal_key in dvh_updated_list[i]['constraints']:
                    df.at[count, 'structure_name'] = dvh_updated_list[i]['parameters']['structure_name']
                    df.at[count, 'dose_gy'] = self.dose_to_gy(dose_key, dvh_updated_list[i]['parameters'][dose_key])
                    df.at[count, 'volume_perc'] = dvh_updated_list[i]['constraints'][goal_key]
                    df.at[count, 'dvh_type'] = 'goal'
                    df.at[count, 'weight'] = dvh_updated_list[i]['parameters']['weight']
                    count = count + 1
            if 'dose_volume_D' in dvh_updated_list[i]['type']:
                limit_key = self.matching_keys(dvh_updated_list[i]['constraints'], 'limit')
                if limit_key in dvh_updated_list[i]['constraints']:
                    df.at[count, 'structure_name'] = dvh_updated_list[i]['parameters']['structure_name']
                    df.at[count, 'volume_perc'] = dvh_updated_list[i]['parameters']['volume_perc']
                    df.at[count, 'dose_gy'] = self.dose_to_gy(limit_key, dvh_updated_list[i]['constraints'][limit_key])
                    df.at[count, 'dvh_type'] = 'constraint'
                    count = count + 1
                goal_key = self.matching_keys(dvh_updated_list[i]['constraints'], 'goal')
                if goal_key in dvh_updated_list[i]['constraints']:
                    df.at[count, 'structure_name'] = dvh_updated_list[i]['parameters']['structure_name']
                    df.at[count, 'volume_perc'] = dvh_updated_list[i]['parameters']['volume_perc']
                    df.at[count, 'dose_gy'] = self.dose_to_gy(goal_key, dvh_updated_list[i]['constraints'][goal_key])
                    df.at[count, 'dvh_type'] = 'goal'
                    df.at[count, 'weight'] = dvh_updated_list[i]['parameters']['weight']
                    count = count + 1
        self.dvh_table = df
        self.get_max_tol(constraints_list=constraint_list)
        self.filter_dvh(my_plan=my_plan)
        return self.dvh_table


    def get_low_dose_vox_ind(self, my_plan: Plan, dose: np.ndarray, inf_matrix: InfluenceMatrix):
        """
        Identifies and stores the indices of low-dose voxels for each DVH constraint or goal.

        For each row in the DVH table, the method:
        - Retrieves the relevant structure name, dose threshold, and volume percentage.
        - Adjusts the volume percentage based on the fraction of the volume in the dose calculation box.
        - Sorts the structure's voxels based on dose values (ascending).
        - Accumulates voxel volumes until the specified volume percentage is reached.
        - Marks these as low-dose voxels and stores them in the `low_dose_voxels` column.

        Parameters
        ----------
        my_plan : Plan
            The treatment plan object that includes the influence matrix.
        dose : np.ndarray
            A 1D array representing the dose values for all optimization voxels.
        inf_matrix : InfluenceMatrix, optional
            The influence matrix that provides voxel-related metadata. If not provided,
            it defaults to `my_plan.inf_matrix`.

        Returns
        -------
        pd.DataFrame
            Updated DVH table with a new column `low_dose_voxels` indicating voxel indices
            in each structure that received the lowest doses contributing up to the specified
            volume percentage.
        """
        dvh_table = self.dvh_table
        inf_matrix = my_plan.inf_matrix
        for ind in dvh_table.index:
            structure_name, dose_gy, vol_perc = dvh_table['structure_name'][ind], dvh_table['dose_gy'][ind], \
            dvh_table['volume_perc'][ind]
            dvh_type = dvh_table['dvh_type'][ind]
            vol_perc = vol_perc / inf_matrix.get_fraction_of_vol_in_calc_box(structure_name)
            struct_vox = inf_matrix.get_opt_voxels_idx(structure_name)
            n_struct_vox = len(struct_vox)
            sort_ind = np.argsort(dose[struct_vox])
            voxel_sort = struct_vox[sort_ind]
            weights = inf_matrix.get_opt_voxels_volume_cc(structure_name)
            weights_sort = weights[sort_ind]
            weight_all_sum = np.sum(weights_sort)
            w_sum = 0
            if dvh_type == 'constraint' or dvh_type == 'goal':
                for w_ind in range(n_struct_vox):
                    w_sum = w_sum + weights_sort[w_ind]
                    w_ratio = w_sum / weight_all_sum
                    if w_ratio * 100 >= (100 - vol_perc):
                        break
                low_dose_voxels = voxel_sort[:w_ind+1]
                if ind == 0:
                    dvh_table.at[ind, 'low_dose_voxels'] = object  # fix issue with adding array to dataframe
                dvh_table.at[ind, 'low_dose_voxels'] = low_dose_voxels

        return self.dvh_table

    def get_max_tol(self, constraints_list: list = None):
        if constraints_list is None:
            constraints_list = self.clinical_criteria_dict['criteria']
        dvh_table = self.dvh_table
        for ind in dvh_table.index:
            structure_name, dose_gy = dvh_table['structure_name'][ind], dvh_table['dose_gy'][ind]
            max_tol = self.get_prescription() * 1.5  # Hard code. Temporary highest value of dose
            for criterion in constraints_list:
                if criterion['type'] == 'max_dose':
                    if criterion['parameters']['structure_name'] == structure_name:
                        limit_key = self.matching_keys(criterion['constraints'], 'limit')
                        if limit_key:
                            max_tol = self.dose_to_gy(limit_key, criterion['constraints'][limit_key])
            dvh_table.at[ind, 'max_tol'] = max_tol

        return self.dvh_table

    def filter_dvh(self, my_plan: Plan):
        dvh_table = deepcopy(self.dvh_table)
        drop_indices = []
        for ind in dvh_table.index:
            structure_name, dose_gy, vol_perc = dvh_table['structure_name'][ind], dvh_table['dose_gy'][ind], \
                dvh_table['volume_perc'][ind]
            vol_perc = vol_perc / my_plan.inf_matrix.get_fraction_of_vol_in_calc_box(structure_name)
            if vol_perc >= 100 or vol_perc <= 0: # remove unnecessary constraints or goals
                drop_indices.append(ind)
        dvh_table = dvh_table.drop(index=drop_indices).reset_index(drop=True)
        self.dvh_table = dvh_table
        return self.dvh_table