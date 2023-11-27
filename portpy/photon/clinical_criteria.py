from typing import List
from .data_explorer import DataExplorer


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

    def __init__(self, data: DataExplorer, protocol_name: str = None, protocol_type: str = 'Default'):
        """

        :param clinical_criteria: dictionary containing information about clinical criteria

        """
        clinical_criteria_dict = data.load_config_clinical_criteria(protocol_name, protocol_type=protocol_type)
        self.clinical_criteria_dict = clinical_criteria_dict

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
