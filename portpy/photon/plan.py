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

from portpy.photon.utils import *
from .ct import CT
from .structures import Structures
from .beam import Beams
from .influence_matrix import InfluenceMatrix
from .clinical_criteria import ClinicalCriteria
try:
    from portpy.photon.vmat_scp import Arcs
except ImportError:
    Arcs = None


class Plan:
    """
    A class representing a plan

    - **Attributes** ::

        :param beams: an object of class Beams that contains information about the beams_dict used in the treatment plan.
        :type beams: object
        :param structures: an object of class Structures that contains information about the structures present in the patient's CT scan.
        :type structures: object
        :param inf_matrix: object of class Influence matrix
        :type inf_matrix: object
        :param clinial_criteria: an object of class ClincialCriteria that contains information about the goals and constraints of the treatment plan
        :type clinical_criteria: object
        :param ct: dictionary containing ct metadata. It includes metadata about important ct parameters like resolution, ct in HU etc.
        :type ct: object
        :param arcs: an object of class Arcs containing information about number of arcs and arcs features
        :type arcs: object

    - **Methods** ::

        :get_prescription():
            Return prescription for the plan
        :get_num_of_fractions()
            Return number of fractions for the plan




    """

    def __init__(self, structs: Structures, beams: Beams, inf_matrix: InfluenceMatrix,
                 ct: CT = None, clinical_criteria: ClinicalCriteria = None, arcs: Arcs = None) -> None:
        """
        Creates an object of Plan class for the specified patient

        :param ct: object of class CT
        :param structs: object of class structures
        :param beams: object of class Beams
        :param inf_matrix: object of class Influence matrix
        :param arcs: object of class Arcs

        :Example:

        >>> my_plan = Plan(structs, beams, inf_matrix, ct, clinical_criteria)
        """

        self.beams: Beams = beams  # create beams attribute
        self.structures: Structures = structs  # create structures attribute
        self.ct: CT = ct  # create ct attribute containing ct information as dictionary
        self.inf_matrix: InfluenceMatrix = inf_matrix
        if clinical_criteria is not None:
            self.clinical_criteria: ClinicalCriteria = clinical_criteria
        if ct is not None:
            self.patient_id = ct.patient_id
        if arcs is not None:
            self.arcs: Arcs = arcs

    def save_plan(self, plan_name: str = None, path: str = None) -> None:
        """

        Save pickled file for plan object

        :param plan_name: create the name of the pickled file of plan object. If none, it will save with the name as 'my_plan'
        :param path: if path is set, plan object will be pickled and saved in path directory else it will save current project directory
        :return: save pickled object of class Plan

        :Example:
        >>> Plan.save_plan(plan_name='my_plan', path=r"path/to/save_plan")
        """
        save_plan(self, plan_name=plan_name, path=path)

    @staticmethod
    def load_plan(plan_name: str = None, path: str = None):
        """
        Load pickle file of the plan object.

        :param plan_name: plan_name of the object of class Plan. It None, it will try to look for plan name called 'my_plan'
        :param path: if path is set, plan object will be load from path directory else current project directory
        :return: load pickled object of class Plan

        :Example:
        >>> Plan.load_plan(plan_name='my_plan', path=r"path/for/loading_plan")
        """

        return load_plan(plan_name=plan_name, path=path)

    @staticmethod
    def load_optimal_sol(sol_name: str, path: str = None) -> dict:
        """

        Load optimal solution dictionary got from optimization

        :param sol_name: name of the optimal solution to be loaded.
        :param path: if path is set, plan object will be load from path directory else current directory
        :return: load solution

        :Example:
        >>> sol = Plan.load_optimal_sol(sol_name='sol', path=r'path/for/loading_sol')
        """
        return load_optimal_sol(sol_name=sol_name, path=path)

    @staticmethod
    def save_optimal_sol(sol: dict, sol_name: str, path: str = None) -> None:
        """
        Save the optimal solution dictionary from optimization

        :param sol: optimal solution dictionary
        :param sol_name: name of the optimal solution saved
        :param path: if path is set, plan object will be load from path directory else current directory
        :return: save pickled file of optimal solution dictionary

        :Example:
        >>> Plan.save_optimal_sol(sol=sol, sol_name='sol', path=r'path/to/save_solution')
        """
        save_optimal_sol(sol=sol, sol_name=sol_name, path=path)

    def get_prescription(self) -> float:
        """

        :return: prescription in Gy
        """
        pres = self.clinical_criteria.clinical_criteria_dict['pres_per_fraction_gy'] * \
               self.clinical_criteria.clinical_criteria_dict[
                   'num_of_fractions']
        return pres

    def get_num_of_fractions(self) -> float:
        """

        :return: number of fractions to be delivered
        """
        return self.clinical_criteria.clinical_criteria_dict['num_of_fractions']

    def get_disease_site(self) -> float:
        """

        :return: number of fractions to be delivered
        """
        return self.clinical_criteria.clinical_criteria_dict['disease_site']

    def save_nrrd(self, sol: dict, data_dir: str = None):
        """
        save nrrd in the path directory else save in patient data directory
        :param sol: optimal solution dict
        :param data_dir: save nrrd images of ct, dose_1d and struct_name set in path directory
        :return: save nrrd images in path
        """
        save_nrrd(self, sol=sol, data_dir=data_dir)