import numpy as np
from typing import List, Union
from portpy.photon.data_explorer import DataExplorer

class Beams:
    """
    A class representing beams_dict.

    - **Attributes** ::

        :param beams_dict: beams_dict dictionary that contains information about the beams_dict in the format of
        dict: {
                   'ID': list(int),
                   'gantry_angle': list(float),
                   'collimator_angle': list(float) }
                  }

        :type beams_dict: dict

    - **Methods** ::

        :get_gantry_angle(beam_id: Optional(int, List[int]):
            Get gantry angle in degrees
        :get_collimator_angle(beam_id):
            Get collimator angle in degrees

    """

    def __init__(self, data: DataExplorer, beam_ids:  List[Union[int, str]] = None, load_inf_matrix_full: bool = False):
        """

        :param beams_dict: Beams dictionary containing information about beams
        """

        metadata = data.load_metadata()
        metadata = self.get_plan_beams(beam_ids=beam_ids, meta_data=metadata)
        beams_dict = data.load_data(meta_data=metadata['beams'], load_inf_matrix_full=load_inf_matrix_full)
        self.beams_dict = beams_dict
        # self.preprocess_beams()

    def get_beamlet_idx_2d_finest_grid(self, beam_id: Union[int, str]) -> np.ndarray:
        """

        :param beam_id: beam_id for the beam
        :return: 2d grid of beamlets in 2.5*2.5 resolution
        """
        ind = self.beams_dict['ID'].index(beam_id)
        return self.beams_dict['beamlet_idx_2d_finest_grid'][ind]

    def get_gantry_angle(self, beam_id: Union[Union[int, str], List[Union[int, str]]]) -> Union[int, List[int]]:
        """
        Get gantry angle

        :param beam_id: beam_id for the beam
        :return: gantry angle for the beam_id
        """
        if isinstance(beam_id, int) or isinstance(beam_id, str):
            ind = self.beams_dict['ID'].index(beam_id)
            return self.beams_dict['gantry_angle'][ind]
        elif isinstance(beam_id, list):
            return [self.beams_dict['gantry_angle'][self.beams_dict['ID'].index(idx)] for idx in beam_id]

    def get_collimator_angle(self, beam_id: Union[Union[int, str], List[Union[int, str]]]) -> Union[int, List[int]]:
        """
        Get collimator angle

        :param beam_id: beam_id for the beam
        :return: collimator angle for the beam_id
        """
        if isinstance(beam_id, int) or isinstance(beam_id, str):
            ind = self.beams_dict['ID'].index(beam_id)
            return self.beams_dict['collimator_angle'][ind]
        elif isinstance(beam_id, list):
            return [self.beams_dict['collimator_angle'][self.beams_dict['ID'].index(idx)] for idx in beam_id]

    def get_jaw_positions(self, beam_id: Union[Union[int, str], List[Union[int, str]]]) -> Union[dict, List[dict]]:
        """
        Get jaw positions

        :param beam_id: beam_id for the beam
        :return: jaw position for the beam_id
        """
        if isinstance(beam_id, int) or isinstance(beam_id, str):
            ind = self.beams_dict['ID'].index(beam_id)
            return self.beams_dict['jaw_position'][ind]
        elif isinstance(beam_id, list):
            return [self.beams_dict['jaw_position'][self.beams_dict['ID'].index(idx)] for idx in beam_id]

    def get_iso_center(self, beam_id: Union[Union[int, str], List[Union[int, str]]]) -> Union[dict, List[dict]]:
        """
        Get iso center for the given beam_id/ids

        :param beam_id: beam_id for the beam
        :return: iso center for the beam_id
        """
        if isinstance(beam_id, int) or  isinstance(beam_id, str):
            ind = self.beams_dict['ID'].index(beam_id)
            return self.beams_dict['iso_center'][ind]
        elif isinstance(beam_id, list):
            return [self.beams_dict['iso_center'][self.beams_dict['ID'].index(idx)] for idx in beam_id]

    def get_spots(self, beam_id: Union[Union[int, str], List[Union[int, str]]]) -> Union[dict, List[dict]]:
        """
        Get jaw positions

        :param beam_id: beam_id for the beam
        :return: jaw position for the beam_id
        """
        if isinstance(beam_id, int) or isinstance(beam_id, str):
            ind = self.beams_dict['ID'].index(beam_id)
            return self.beams_dict['spots'][ind]
        elif isinstance(beam_id, list):
            return [self.beams_dict['spots'][self.beams_dict['ID'].index(idx)] for idx in beam_id]

    @staticmethod
    def get_plan_beams(beam_ids: List[Union[int, str]] = None, meta_data: dict = None) -> dict:
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
            if 'planner_beam_ids' in meta_data:
                beam_ids = meta_data['planner_beam_ids']['IDs']
            else:
                beam_ids = meta_data['beams']['ID']
        meta_data_req = meta_data.copy()
        del meta_data_req['beams']  # remove previous beams_dict
        beamReq = dict()
        for i in range(len(beam_ids)):
            if beam_ids[i] in meta_data['beams']['ID']:
                ind = meta_data['beams']['ID'].index(beam_ids[i])
                for key in meta_data['beams']:
                    beamReq.setdefault(key, []).append(meta_data['beams'][key][ind])
            else:
                print('beam id {} is not available'.format(beam_ids[i]))
        meta_data_req['beams'] = beamReq
        return meta_data_req
