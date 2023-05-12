from typing import List
from portpy.photon.beam import Beams
from portpy.photon.structures import Structures
from portpy.photon.influence_matrix import InfluenceMatrix
from portpy.photon.utils import *
from .data_explorer import DataExplorer
from .ct import CT
from .structures import Structures
from .beam import Beams
from .influence_matrix import InfluenceMatrix
from .clinical_criteria import ClinicalCriteria


class Plan:
    """
    A class representing a plan

    - **Attributes** ::

        :param opt_beamlets_PTV_margin_mm: For each beam, we often only include the beamlets that are within
            few millimetres of the projection of the PTV (tumor) into that beam. It is because the beamlets
            that are far from the PTV projection mainly deliver radiation to the healthy tissues not PTV. Default is 3mm
        :type opt_beamlets_PTV_margin_mm: int
        :param beams: an object of class Beams that contains information about the beams_dict used in the treatment plan.
        :type beams: object
        :param structures: an object of class Structures that contains information about the structures present in the patient's CT scan.
        :type structures: object
        :param inf_matrix: object of class Influence matrix
        :type inf_matrix: object
        :param clinial_criteria: an object of class ClincialCriteria that contains information about the goals and constraints of the treatment plan
        :type inf_matrix: object
        :param ct: dictionary containing ct metadata. It includes metadata about important ct parameters like resolution, ct in HU etc.
        :type ct: object

    - **Methods** ::

        :create_inf_matrix(beamlet_width_mm, beamlet_height_mm,
                          down_sample_xyz):
            Create object of Influence Matrix class
        :get_prescription():
            Return prescription for the plan
        :get_num_of_fractions()
            Return number of fractions for the plan




    """

    def __init__(self, ct: CT, structs: Structures, beams: Beams, inf_matrix: InfluenceMatrix,
                 clinical_criteria: ClinicalCriteria) -> None:
        """
        Creates an object of Plan class for the specified patient

        :param ct: object of class CT
        :param structs: object of class structures
        :param beams: object of class Beams
        :param inf_matrix: object of class Influence matrix

        :Example:

        >>> my_plan = Plan(ct, structs, beams, inf_matrix, clinical_criteria)
        """

        self.beams = beams  # create beams attribute
        self.structures = structs  # create structures attribute
        self.ct = ct  # create ct attribute containing ct information as dictionary
        self.inf_matrix = inf_matrix
        self.clinical_criteria = clinical_criteria

    def save_plan(self, plan_name: str = None, path: str = None) -> None:
        """

        Save pickled file for plan object

        :param plan_name: create the name of the pickled file of plan object. If none, it will save with the name as 'my_plan'
        :param path: if path is set, plan object will be pickled and saved in path directory else it will save current project directory
        :return: save pickled object of class Plan

        :Example:
        >>> my_plan.save_plan(plan_name='my_plan', path=r"path/to/save_plan")
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
        >>> my_plan.save_optimal_sol(sol=sol, sol_name='sol', path=r'path/to/save_solution')
        """
        save_optimal_sol(sol=sol, sol_name=sol_name, path=path)

    def create_inf_matrix(self, beamlet_width_mm: float = 2.5, beamlet_height_mm: float = 2.5,
                          opt_vox_xyz_res_mm: List[float] = None,
                          structure: str = 'PTV', is_full: bool = False) -> InfluenceMatrix:
        """
                Create a influence matrix object for Influence Matrix class

                :param is_full: Defaults to False. If True, it will create full influence matrix
                :param beamlet_width_mm: beamlet width in mm. It should be multiple of 2.5, defaults to 2.5
                :param beamlet_height_mm: beamlet height in mm. It should be multiple of 2.5, defaults to 2.5
                :param structure: target struct_name for creating BEV beamlets, defaults to 'PTV'
                :param opt_vox_xyz_res_mm: It down-samples optimization voxels as factor of ct resolution
                    e.g. opt_vox_xyz_res = [5*ct.res.x,5*ct.res.y,1*ct.res.z]. It will down-sample optimization voxels with 5 * ct res. in x direction, 5 * ct res. in y direction and 1*ct res. in z direction.
                    defaults to None. When None it will use the original optimization voxel resolution.
                :returns: object of influence Matrix class

                :Example:
                >>> inf_matrix = my_plan.create_inf_matrix(beamlet_width_mm=5, beamlet_height_mm=5, opt_vox_xyz_res_mm=[5,5,1], struct_name=struct_name)
                """
        return InfluenceMatrix(self, beamlet_width_mm=beamlet_width_mm, beamlet_height_mm=beamlet_height_mm,
                               opt_vox_xyz_res_mm=opt_vox_xyz_res_mm, is_full=is_full)

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

    def add_rinds(self, rind_params: List[dict], inf_matrix=None):
        """
        Example for
        rind_params = [{'rind_name': 'RIND_0', 'ref_structure': 'PTV, 'margin_start_mm': 2, 'margin_end_mm': 10, 'max_dose_gy': 10}]

        :param rind_params: rind_params as dictionary
        :param inf_matrix: object of class inf_matrix
        :return: save rinds to plan object
        """

        if inf_matrix is None:
            inf_matrix = self.inf_matrix
        print('creating rinds..')

        ct_to_dose_map = inf_matrix.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        dose_mask = ct_to_dose_map >= 0
        dose_mask = dose_mask.astype(int)
        self.structures.create_structure('dose_mask', dose_mask)

        for ind, param in enumerate(rind_params):
            rind_name = param['rind_name']
            first_dummy_name = '{}_{}'.format(param['ref_structure'], param['margin_start_mm'])
            second_dummy_name = '{}_{}'.format(param['ref_structure'], param['margin_end_mm'])
            self.structures.expand(param['ref_structure'], margin_mm=param['margin_start_mm'],
                                   new_structure=first_dummy_name)
            if param['margin_end_mm'] == 'inf':
                param['margin_end_mm'] = 500
            self.structures.expand(param['ref_structure'], margin_mm=param['margin_end_mm'],
                                   new_structure=second_dummy_name)
            self.structures.subtract(second_dummy_name, first_dummy_name, str1_sub_str2=rind_name)
            self.structures.delete_structure(first_dummy_name)
            self.structures.delete_structure(second_dummy_name)
            self.structures.intersect(rind_name, 'dose_mask', str1_and_str2=rind_name)
        self.structures.delete_structure('dose_mask')

        print('rinds created!!')

        for param in rind_params:
            inf_matrix.set_opt_voxel_idx(self, structure_name=param['rind_name'])
            # add rind constraint
            parameters = {'structure_name': param['rind_name']}
            # total_pres = self.get_prescription()
            if 'max_dose_gy' in param:
                constraints = {'limit_dose_gy': param['max_dose_gy']}
                self.clinical_criteria.add_criterion(criterion='max_dose', parameters=parameters,
                                                     constraints=constraints)
            if 'mean_dose_gy' in param:
                constraints = {'limit_dose_gy': param['mean_dose_gy']}
                self.clinical_criteria.add_criterion(criterion='mean_dose', parameters=parameters,
                                                     constraints=constraints)

    def save_nrrd(self, sol: dict, data_dir: str = None):
        """
        save nrrd in the path directory else save in patient data directory
        :param sol: optimal solution dict
        :param data_dir: save nrrd images of ct, dose_1d and struct_name set in path directory
        :return: save nrrd images in path
        """
        save_nrrd(self, sol=sol, data_dir=data_dir)