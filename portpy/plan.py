import numpy as np
from portpy import load_metadata, load_data
from .beam import Beams
from .structures import Structures
import os
from portpy.clinical_criteria import ClinicalCriteria
from portpy.influence_matrix import InfluenceMatrix
import SimpleITK as sitk
from portpy.visualization import Visualization
from portpy.optimization import Optimization

# from typing import Dict, List, Optional, Union
import pickle


class Plan:
    """
    A class representing plan.
    """

    def __init__(self, patient_id, data_folder=None, beam_ids=None, opt_beamlets_PTV_margin_mm=None, options=None):

        # super().__init__()
        if data_folder is None:
            data_folder = os.path.join(os.getcwd(), "..", 'Data')
        patient_folder_path = os.path.join(data_folder, patient_id)

        if opt_beamlets_PTV_margin_mm is None:
            self.opt_beamlets_PTV_margin_mm = 3
        # read all the meta data for the required patient
        meta_data = load_metadata(patient_folder_path, options=options)

        # options for loading requested data
        # if 1 then load the data. if 0 then skip loading the data
        # meta_data = self.load_options(options=options, meta_data=meta_data)
        # filter metadata for the given beam_indices
        meta_data = self.get_plan_beams(beam_ids=beam_ids, meta_data=meta_data)
        # Load data
        data = load_data(meta_data, patient_folder_path)
        self.beams = Beams(data['beams'], opt_beamlets_PTV_margin_mm=opt_beamlets_PTV_margin_mm)
        self.structures = Structures(data['structures'], data['opt_voxels'])
        self.ct = data['ct']
        # self.structures.create_rinds()
        self.patient_id = patient_id
        self.clinical_criteria = ClinicalCriteria(data['clinical_criteria'])
        self.inf_matrix = InfluenceMatrix(self)
        # self.opt_sol = []

    @staticmethod
    def get_plan_beams(beam_ids=None, meta_data=None):
        if beam_ids is None:
            beam_ids = meta_data['planner_beam_ids']['IDs']
        meta_data_req = meta_data.copy()
        del meta_data_req['beams']
        beamReq = dict()
        inds = []
        for i in range(len(beam_ids)):
            if beam_ids[i] in meta_data['beams']['ID']:
                ind = np.where(np.array(meta_data['beams']['ID']) == beam_ids[i])
                ind = ind[0][0]
                inds.append(ind)
                for key in meta_data['beams']:
                    beamReq.setdefault(key, []).append(meta_data['beams'][key][ind])
        meta_data_req['beams'] = beamReq
        if len(inds) < len(beam_ids):
            print('some indices are not available')
        return meta_data_req

    @staticmethod
    def load_options(options=None, meta_data=None):
        if options is None:
            options = dict()
            options['loadInfluenceMatrixFull'] = 0
            options['loadInfluenceMatrixSparse'] = 1
        if len(options) != 0:
            if 'loadInfluenceMatrixFull' in options and not options['loadInfluenceMatrixFull']:
                meta_data['beams']['influenceMatrixFull_File'] = [None] * len(
                    meta_data['beams']['influenceMatrixSparse_File'])
            if 'loadInfluenceMatrixSparse' in options and not options['loadInfluenceMatrixSparse']:
                meta_data['beams']['influenceMatrixFull_File'] = [None] * len(
                    meta_data['beams']['influenceMatrixSparse_File'])
        return meta_data

    def save_nrrd(self, sol=None, path=None):
        """
        save nrrd in the path directory else save in patient data directory
        :param sol: optimal solution dict
        :param path: save nrrd images of ct, dose_1d and structure set in path directory
        :return:
        """
        import os
        if path is None:
            path = os.path.join(os.getcwd(), "..", 'Data', self.patient_id)
        # scale_mat = np.ones(3)
        # scale_mat[0, 0] = 1 / self.ct['resolution_xyz_mm'][0]
        # scale_mat[1, 1]  = 1 / self.ct['resolution_xyz_mm'][1]
        # scale_mat[2, 2] = 1 / self.ct['resolution_xyz_mm'][2]
        # ijkMat = scale_mat * affineMat(1:3, 1: 3)
        ct_arr = self.ct['ct_hu_3d'][0]
        ct = sitk.GetImageFromArray(ct_arr)
        ct.SetOrigin(self.ct['origin_xyz_mm'])
        ct.SetSpacing(self.ct['resolution_xyz_mm'])
        ct.SetDirection(self.ct['direction'])
        sitk.WriteImage(ct, os.path.join(path, 'ct.nrrd'))

        if sol['inf_matrix'].dose_3d is None:
            dose_arr = sol['inf_matrix'].dose_1d_to_3d(sol=sol)
        else:
            dose_arr = sol['inf_matrix'].dose_3d
        dose = sitk.GetImageFromArray(dose_arr)
        dose.SetOrigin(self.ct['origin_xyz_mm'])
        dose.SetSpacing(self.ct['resolution_xyz_mm'])
        dose.SetDirection(self.ct['direction'])
        sitk.WriteImage(dose, os.path.join(path, 'dose.nrrd'))

        labels = self.structures.structures_dict['structure_mask_3d']
        mask_arr = np.array(labels).transpose((1, 2, 3, 0))
        mask = sitk.GetImageFromArray(mask_arr.astype('uint8'))
        # for i, struct_name in enumerate(self.structures.structures_dict['name']):
        #     segment_name = "Segment{0}_Name".format(i)
        #     mask.SetMetaData(segment_name, struct_name)
        mask.SetOrigin(self.ct['origin_xyz_mm'])
        mask.SetSpacing(self.ct['resolution_xyz_mm'])
        mask.SetDirection(self.ct['direction'])
        sitk.WriteImage(mask, os.path.join(path, 'rtss.seg.nrrd'), True)
        # self.visualize.patient_name = self.patient_name

    def save_plan(self, plan_name=None, path=None):
        """
        :param plan_name: create the name of the pickled file of plan object
        :param path: if path is set, plan object will be pickled and saved in path directory else current directory
        :return: save pickled object of class Plan
        """
        if path is None:
            path = os.getcwd()
        elif not os.path.exists(path):
            os.makedirs(path)

        if plan_name is None:
            plan_name = 'my_plan'
        pickle_file = open(os.path.join(path, plan_name), 'wb')
        # pickle the dictionary and write it to file
        pickle.dump(self, pickle_file)
        # close the file
        pickle_file.close()

    @staticmethod
    def load_plan(plan_name=None, path=None):
        """
        :param plan_name: plan_name of the object of class Plan
        :param path: if path is set, plan object will be load from path directory else current directory
        :return: load pickled object of class Plan
        """
        if path is None:
            path = os.getcwd()
        elif not os.path.exists(path):
            os.makedirs(path)

        if plan_name is None:
            plan_name = 'my_plan'
        pickle_file = open(os.path.join(path, plan_name), 'rb')
        my_plan = pickle.load(pickle_file)
        return my_plan

    @staticmethod
    def load_optimal_sol(sol_name: str, path=None):
        """
        :param sol_name: name of the optimal solution saved
        :param path: if path is set, plan object will be load from path directory else current directory
        :return: load pickled object of class Plan
        """
        if path is None:
            path = os.getcwd()
        elif not os.path.exists(path):
            os.makedirs(path)

        pickle_file = open(os.path.join(path, sol_name), 'rb')
        my_plan = pickle.load(pickle_file)
        return my_plan

    @staticmethod
    def save_optimal_sol(sol, sol_name, path=None):
        """
        :param sol: optimal solution dictionary
        :param sol_name: name of the optimal solution saved
        :param path: if path is set, plan object will be load from path directory else current directory
        :return: save pickled file of optimal solution dictionary
        """
        if path is None:
            path = os.getcwd()
        elif not os.path.exists(path):
            os.makedirs(path)
        pickle_file = open(os.path.join(path, sol_name), 'wb')
        # pickle the dictionary and write it to file
        pickle.dump(sol, pickle_file)

    def create_inf_matrix(self, beamlet_width=2.5, beamlet_height=2.5, down_sample_xyz=None):

        return InfluenceMatrix(self, beamlet_width=beamlet_width, beamlet_height=beamlet_height, down_sample_xyz=down_sample_xyz)

    def get_prescription(self):
        pres = self.clinical_criteria.clinical_criteria_dict['pres_per_fraction_gy'] * \
               self.clinical_criteria.clinical_criteria_dict[
                   'num_of_fractions']
        return pres

    def get_num_of_fractions(self):
        return self.clinical_criteria.clinical_criteria_dict['num_of_fractions']

    def plot_dvh(self, sol=None, dose_1d=None, structs=None, options_norm=None, options_fig=None):
        Visualization.plot_dvh(self, sol=sol, dose_1d=dose_1d, structs=structs, options_norm=options_norm, options_fig=options_fig)

    def run_IMRT_fluence_map_CVXPy(self, inf_matrix=None, solver='MOSEK'):
        Optimization.run_IMRT_fluence_map_CVXPy(self, inf_matrix=inf_matrix, solver=solver)

    @staticmethod
    def plot_fluence_2d(beam_id: int, sol: dict = None):
        Visualization.plot_fluence_2d(beam_id=beam_id, sol=sol)

    @staticmethod
    def plot_fluence_3d(beam_id: int, sol: dict = None):
        Visualization.plot_fluence_3d(beam_id=beam_id, sol=sol)

    def view_in_slicer(self, slicer_path=None, img_dir=None):
        Visualization.view_in_slicer(self, slicer_path=slicer_path, img_dir=img_dir)
