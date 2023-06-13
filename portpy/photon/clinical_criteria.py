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
                      "name": "max_dose",
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

    def add_criterion(self, criterion: str, parameters: dict, constraints: dict) -> None:
        """
        Add criterion to the clinical criteria dictionary

        :param criterion: criterion name. e.g. max_dose
        :param parameters: parameters dictionary e.g. parameters = {'struct':'PTV'}
        :param constraints: constraints dictionary e.g. constraints = {'limit_dose_gy':66, 'goal_dose_gy':60}
        :return: add the criteria to clinical criteria dictionary

        """

        self.clinical_criteria_dict['criteria'].append({'name': criterion})
        self.clinical_criteria_dict['criteria'][-1]['parameters'] = parameters
        self.clinical_criteria_dict['criteria'][-1]['constraints'] = constraints

    def modify_criterion(self, criterion: str, parameters: dict, constraints: dict) -> None:
        """

        Modify the criterion in clinical criteria dictionary

        :param criterion: criterion name. e.g. max_dose
        :param parameters: parameters dictionary e.g. parameters = {'struct':'PTV'}
        :param constraints: constraints dictionary e.g. constraints = {'limit_dose_gy':66, 'goal_dose_gy':60}
        :return: add the criteria to clinical criteria dictionary

        """
        criteria = self.clinical_criteria_dict['criteria']
        ind = [criteria[i]['name'] for i in range(len(criteria)) if criteria[i]['name'] == criterion and
               criteria[i]['parameters'] == parameters]
        if len(ind) > 0:

            criteria[ind[0]]['constraints'] = constraints
        else:
            raise Exception('No criteria  for name {}  and parameters {}'.format(criterion, parameters))

    @staticmethod
    def create_criterion(criterion: str, parameters: dict, constraints: dict):
        """
        Create criterion and return list

        :param criterion: criterion name. e.g. max_dose
        :param parameters: parameters dictionary e.g. parameters = {'struct':'PTV'}
        :param constraints: constraints dictionary e.g. constraints = {'limit_dose_gy':66, 'goal_dose_gy':60}
        :return: add the criteria to clinical criteria dictionary

        """

        criterion = [{'name': criterion, 'parameters': parameters, 'constraints': constraints}]
        return criterion

    def get_criteria(self, name: str = None) -> List[dict]:
        """
        Returns all the clinical criteria
        :return:
        """
        all_criteria = []
        if name is None:
            all_criteria = self.clinical_criteria_dict['criteria']
        elif name == 'max_dose':
            criteria = self.clinical_criteria_dict['criteria']
            ind = [i for i in range(len(criteria)) if criteria[i]['name'] == name]
            if len(ind) > 0:
                all_criteria = [criteria[i] for i in ind]
        elif name == 'mean_dose':
            criteria = self.clinical_criteria_dict['criteria']
            ind = [i for i in range(len(criteria)) if criteria[i]['name'] == name]
            if len(ind) > 0:
                all_criteria = [criteria[i] for i in ind]

        if isinstance(all_criteria, dict):
            all_criteria = [all_criteria]
        return all_criteria
