import numpy as np
import matplotlib.pyplot as plt


class ClinicalCriteria:
    """
    A class representing clinical criteria.
    """

    def __init__(self, clinical_criteria):
        # self.optimal_intensity = None
        self.optimal_intensity = None
        self.clinical_criteria_dict = clinical_criteria

    def add_criterion(self, criterion=None, parameters=None, constraints=None):

        self.clinical_criteria_dict['criteria'].append({'name': criterion})
        self.clinical_criteria_dict['criteria'][-1]['parameters'] = parameters
        self.clinical_criteria_dict['criteria'][-1]['constraints'] = constraints


