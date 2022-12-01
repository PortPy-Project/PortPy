import numpy as np
from portpy import load_metadata, load_data
from .beam import Beams
from .structures import Structures
import os
from portpy.visualization import Visualization
from portpy.optimization import Optimization
from portpy.clinical_criteria import ClinicalCriteria
from portpy.evaluation import Evaluation


# from typing import Dict, List, Optional, Union


class Plan:
    """
    A class representing plan.
    """

    def __init__(self, patient_name, beam_ids=None, options=None, eval=True, visual=True):

        # super().__init__()
        patient_folder_path = os.path.join(os.getcwd(), "..", 'Data', patient_name)

        # read all the meta data for the required patient
        meta_data = load_metadata(patient_folder_path)

        # options for loading requested data
        # if 1 then load the data. if 0 then skip loading the data
        meta_data = self.load_options(options=options, meta_data=meta_data)
        # filter metadata for the given beam_indices
        my_plan = self.get_plan_beams(beam_ids=beam_ids, meta_data=meta_data)
        # Load data
        my_plan = load_data(my_plan, my_plan['patient_folder_path'])
        self.beams = Beams(my_plan['beams'])
        self.structures = Structures(my_plan['structures'], my_plan['opt_voxels'])
        self.ct = my_plan['ct']
        # self.structures.create_rinds()
        self.patient_name = patient_name
        self.clinical_criteria = ClinicalCriteria(my_plan['clinical_criteria'])
        self.optimize = Optimization(beams=self.beams, structures=self.structures,
                                     clinical_criteria=self.clinical_criteria)
        if eval:
            self.evaluate = Evaluation(beams=self.beams, structures=self.structures,
                                       clinical_criteria=self.clinical_criteria)
        if visual:
            self.visualize = Visualization(beams=self.beams, structures=self.structures,
                                           clinical_criteria=self.clinical_criteria, evaluate=self.evaluate, ct=self.ct)

    @staticmethod
    def get_plan_beams(beam_ids=None, meta_data=None):
        if beam_ids is None:
            beam_ids = meta_data['planner_beam_ids']['IDs']
        my_plan = meta_data.copy()
        del my_plan['beams']
        beamReq = dict()
        inds = []
        for i in range(len(beam_ids)):
            if beam_ids[i] in meta_data['beams']['ID']:
                ind = np.where(np.array(meta_data['beams']['ID']) == beam_ids[i])
                ind = ind[0][0]
                inds.append(ind)
                for key in meta_data['beams']:
                    beamReq.setdefault(key, []).append(meta_data['beams'][key][ind])
        my_plan['beams'] = beamReq
        if len(inds) < len(beam_ids):
            print('some indices are not available')
        return my_plan

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
