import numpy as np
from typing import List

from portpy.utils import *
from portpy.beam import Beams
from portpy.structures import Structures
import os
from portpy.clinical_criteria import ClinicalCriteria
from portpy.influence_matrix import InfluenceMatrix
from portpy.visualization import Visualization
from portpy.optimization import Optimization
from pathlib import Path
# from typing import Dict, List, Optional, Union
# from functools import wraps
import pickle


class Plan:
    """
    A class representing a plan

    - **Attributes** ::

        :param opt_beamlets_PTV_margin_mm: For each beam, we often only include the beamlets that are within
            few millimetres of the projection of the PTV (tumor) into that beam. It is because the beamlets
            that are far from the PTV projection mainly deliver radiation to the healthy tissues not PTV. Default is 3mm
        :type opt_beamlets_PTV_margin_mm: int
        :param beams_dict: an object of class Beams that contains information about the beams_dict used in the treatment plan.
        :type beams: object
        :param structures: an object of class Structures that contains information about the structures present in the patient's CT scan.
        :type structures: object
        :param inf_matrix: object of class Influence matrix
        :type inf_matrix: object
        :param clinial_criteria: an object of class ClincialCriteria that contains information about the goals and constraints of the treatment plan
        :type inf_matrix: object
        :param ct: dictionary containing ct metadata. It includes metadata about important ct parameters like resolution, ct in HU etc.
        :type ct: dict

    - **Methods** ::

        :create_inf_matrix(beamlet_width_mm, beamlet_height_mm,
                          down_sample_xyz):
            Create object of Influence Matrix class
        :get_prescription():
            Return prescription for the plan
        :get_num_of_fractions()
            Return number of fractions for the plan




    """

    def __init__(self, patient_id: str, data_dir: str = None, beam_ids: List[int] = None,
                 opt_beamlets_PTV_margin_mm: int = 3, load_inf_matrix_full: bool = False) -> None:
        """
        Creates an object of Plan class for the specified patient

        :param patient_id: the patient that we're creating a plan for (name of the folder)
        :param data_dir: the name of the folder that includes all the patients' data.
            If None, then ''current-folder\Data'' is used
        :param beam_ids: the indices of the beams_dict used for creating the plan object.
            If None, the beams_dict selected by an expert (human) physicist for the specified patient would be used
        :param opt_beamlets_PTV_margin_mm: For each beam, we often only include the beamlets that are within
            few millimetres of the projection of the PTV (tumor) into that beam. It is because the beamlets
            that are far from the PTV projection mainly deliver radiation to the healthy tissues not PTV. Default is 3mm
        :param load_inf_matrix_full: If set to true, it will load full influence matrix from the data

        :Example:

        >>> my_plan = Plan(patient_id = r"Lung_Patient_1", path = r"c:\Data", beam_ids = [0,1,2,3,4,5,6], opt_beamlets_PTV_margin_mm=3)
        """

        if data_dir is None:
            data_dir = os.path.join(Path(__file__).parents[1], 'Data')
        patient_folder_path = os.path.join(data_dir, patient_id)

        # read all the meta data for the specified patient
        meta_data = load_metadata(patient_folder_path)

        # options for loading requested data
        # if 1 then load the data. if 0 then skip loading the data
        # meta_data = my_plan.load_options(options=options, meta_data=meta_data)
        # filter metadata for the given beam_indices.
        # The meta_data originally includes all the beams_dict.  get_plan_beams only keeps the requested beams_dict
        meta_data = self.get_plan_beams(beam_ids=beam_ids, meta_data=meta_data)
        # Load all the data (e.g., influence matrix, voxel coordinates).
        # meta_data does not include the large numeric data stored in .h5 files
        data = load_data(meta_data=meta_data, pat_dir=patient_folder_path, load_inf_matrix_full=load_inf_matrix_full)
        self.opt_beamlets_PTV_margin_mm = opt_beamlets_PTV_margin_mm
        self.beams = Beams(data['beams_dict'])  # create beams_dict object
        self.structures = Structures(data['structures'], data['opt_voxels'])  # create structures object
        self.ct = data['ct']  # create ct attribute containing ct information as dictionary
        self.patient_id = patient_id
        self.clinical_criteria = ClinicalCriteria(data['clinical_criteria'])  # create clinical criteria object
        is_full = False
        if load_inf_matrix_full:  # check if full influence matrix is requested
            is_full = True

        self.inf_matrix = InfluenceMatrix(self, is_full=is_full)  # create influence matrix object

    @staticmethod
    def get_plan_beams(beam_ids: List[int] = None, meta_data: dict = None) -> dict:
        """
        Create and return a copy of meta_data with only including the requested beams_dict (beam_ids)


        :param beam_ids: the indices of the beams_dict to be included. If None, the planner's beams_dict are used
        :param meta_data: the dictionary including all the beams_dict
        :return: returns the meta_data dictionary only including the requested beams_dict in format of:
            dict: {
                   'structures': {'name': list(str), 'volume_cc': list(float), }
                   'opt_voxels': {'_ct_voxel_resolution_xyz_mm': list(float),}
                  }
        """
        if beam_ids is None:  # if beam_ids not included, then the beams_dict
            # selected by an expert human planner would be used
            beam_ids = meta_data['planner_beam_ids']['IDs']
        meta_data_req = meta_data.copy()
        del meta_data_req['beams_dict']  # remove previous beams_dict
        beamReq = dict()
        for i in range(len(beam_ids)):
            if beam_ids[i] in meta_data['beams_dict']['ID']:
                ind = meta_data['beams_dict']['ID'].index(beam_ids[i])
                for key in meta_data['beams_dict']:
                    beamReq.setdefault(key, []).append(meta_data['beams_dict'][key][ind])
            else:
                print('beam id {} is not available'.format(beam_ids[i]))
        meta_data_req['beams_dict'] = beamReq
        return meta_data_req

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
                :param structure: target structure for creating BEV beamlets, defaults to 'PTV'
                :param opt_vox_xyz_res_mm: It down-samples optimization voxels as factor of ct resolution
                    e.g. opt_vox_xyz_res = [5*ct.res.x,5*ct.res.y,1*ct.res.z]. It will down-sample optimization voxels with 5 * ct res. in x direction, 5 * ct res. in y direction and 1*ct res. in z direction.
                    defaults to None. When None it will use the original optimization voxel resolution.
                :returns: object of influence Matrix class

                :Example:
                >>> inf_matrix = my_plan.create_inf_matrix(beamlet_width_mm=5, beamlet_height_mm=5, opt_vox_xyz_res_mm=[5,5,1], structure=structure)
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

    def get_ct_res_xyz_mm(self) -> List[float]:
        """

        :return: number of fractions to be delivered
        """
        return self.ct['resolution_xyz_mm']

    def plot_dvh(self, sol, dose_1d=None, structs=None, dose_scale: Visualization.dose_type = "Absolute(Gy)",
                 volume_scale: Visualization.volume_type = "Relative(%)", **options):
        """
                Create dvh plot for the selected structures

                :param sol: optimal sol dictionary
                :param dose_1d: dose_1d in 1d voxels
                :param structs: structures to be included in dvh plot
                :param volume_scale: volume scale on y-axis. Default= Absolute(cc). e.g. volume_scale = "Absolute(cc)" or volume_scale = "Relative(%)"
                :param dose_scale: dose_1d scale on x axis. Default= Absolute(Gy). e.g. dose_scale = "Absolute(Gy)" or dose_scale = "Relative(%)"
                :keyword style (str): line style for dvh curve. default "solid". can be "dotted", "dash-dotted".
                :keyword width (int): width of line. Default 2
                :keyword colors(list): list of colors
                :keyword legend_font_size: Set legend_font_size. default 10
                :keyword figsize: Set figure size for the plot. Default figure size (12,8)
                :keyword create_fig: Create a new figure. Default True. If False, append to the previous figure
                :keyword title: Title for the figure
                :keyword filename: Name of the file to save the figure in current directory
                :keyword show: Show the figure. Default is True. If false, next plot can be append to it
                :keyword norm_flag: Use to normalize the plan. Default is False.
                :keyword norm_volume: Use to set normalization volume. default is 90 percentile.
                :return: dvh plot for the selected structures
                """

        Visualization.plot_dvh(self, sol=sol, dose_1d=dose_1d, structs=structs, dose_scale=dose_scale,
                               volume_scale=volume_scale, **options)

    def run_IMRT_fluence_map_CVXPy(self, inf_matrix: InfluenceMatrix = None, solver='MOSEK'):
        Optimization.run_IMRT_fluence_map_CVXPy(self, inf_matrix=inf_matrix, solver=solver)

    @staticmethod
    def plot_fluence_2d(beam_id: int, sol: dict = None):
        """
        plot fluence in 2d for beam_id
        :param beam_id: beam_id of the beam
        :param sol: solution dictionary after optimization
        :return: 2d optimal fluence plot
        """
        Visualization.plot_fluence_2d(beam_id=beam_id, sol=sol)

    @staticmethod
    def plot_fluence_3d(beam_id: int, sol: dict = None):
        """
        plot fluence in 3d for beam_id
        :param sol: solution after optimization
        :param beam_id: beam_id of the beam
        :return: 3d optimal fluence plot
        """
        Visualization.plot_fluence_3d(beam_id=beam_id, sol=sol)

    def save_nrrd(self, sol: dict, data_dir: str = None):
        """
        save nrrd in the path directory else save in patient data directory
        :param sol: optimal solution dict
        :param data_dir: save nrrd images of ct, dose_1d and structure set in path directory
        :return: save nrrd images in path
        """
        save_nrrd(self, sol=sol, data_dir=data_dir)

    def view_in_slicer(self, slicer_path=None, img_dir=None):
        """
        view ct, dose_1d and structures in slicer
        :param slicer_path:
        :param img_dir:
        :return:
        """
        Visualization.view_in_slicer(self, slicer_path=slicer_path, data_dir=img_dir)
