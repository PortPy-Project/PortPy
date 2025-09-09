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

import numpy as np
from typing import List, Union
from .data_explorer import DataExplorer


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
        self.preprocess_beams()
        self.patient_id = data.patient_id

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

    def get_beamlets(self, beam_id: Union[Union[int, str], List[Union[int, str]]]) -> Union[dict, List[dict]]:
        """
        Get jaw positions

        :param beam_id: beam_id for the beam
        :return: jaw position for the beam_id
        """
        if isinstance(beam_id, int) or isinstance(beam_id, str):
            ind = self.beams_dict['ID'].index(beam_id)
            return self.beams_dict['beamlets'][ind]
        elif isinstance(beam_id, list):
            return [self.beams_dict['beamlets'][self.beams_dict['ID'].index(idx)] for idx in beam_id]

    @staticmethod
    def sort_beamlets(b_map):
        c = b_map[b_map >= 0]
        c = np.unique(c)
        ind = np.arange(0, len(c))
        c_sort = np.sort(c)
        matrix_ind = [np.where(b_map == c_i) for c_i in c_sort]
        map_copy = b_map.copy()
        for i in range(len(ind)):
            map_copy[matrix_ind[i]] = ind[i]
        return map_copy

    def make_beamlets_continous(self):

        # make beamlets continuous

        for ind, id in enumerate(self.beams_dict['ID']):
            beam_map = self.get_beamlet_idx_2d_finest_grid(beam_id=id)
            std_map = self.sort_beamlets(beam_map)
            if ind == 0:
                self.beams_dict['beamlet_idx_2d_finest_grid'][ind] = std_map
            else:

                beamlet_ind = np.unique(std_map.flatten())
                beamlet_ind = beamlet_ind[beamlet_ind >= 0]
                a = np.where(np.isin(std_map, beamlet_ind))
                mask_2d_grid = np.zeros_like(std_map, dtype=bool)
                mask_2d_grid[a] = True
                beam_map = std_map + int(
                    np.amax(self.beams_dict['beamlet_idx_2d_finest_grid'][ind - 1]) + 1) * mask_2d_grid
                self.beams_dict['beamlet_idx_2d_finest_grid'][ind] = beam_map

    def preprocess_beams(self):
        for i, beam_id in enumerate(self.beams_dict['ID']):
            ind = self.beams_dict['ID'].index(beam_id)
            beam_2d_grid = self.create_beamlet_idx_2d_finest_grid(beam_id=beam_id)
            self.beams_dict.setdefault('beamlet_idx_2d_finest_grid', []).append(beam_2d_grid)

    @staticmethod
    def _create_2d_orig_grid(beam_map):
        rowsNoRepeat = [0]
        for i in range(1, np.size(beam_map, 0)):
            if (beam_map[i, :] != beam_map[rowsNoRepeat[-1], :]).any():
                rowsNoRepeat.append(i)
        colsNoRepeat = [0]
        for j in range(1, np.size(beam_map, 1)):
            if (beam_map[:, j] != beam_map[:, colsNoRepeat[-1]]).any():
                colsNoRepeat.append(j)
        beam_map = beam_map[np.ix_(np.asarray(rowsNoRepeat), np.asarray(colsNoRepeat))]
        return beam_map

    def get_beamlet_idx_2d_grid(self, beam_id: Union[Union[int, str], List[Union[int, str]]] = None, ind: int = None,
                                finest_grid: bool = False) -> Union[np.ndarray, List[np.ndarray]]:

        """
        get BEV in 2d grid in original resolution where beamlet index is the column number in influence matrix

        :param ind: idx of the beam in beamlets_dict.
        It can be int or List[int]. If int, ndarray is return, If List[int], list[ndarray] is returned
        :param beam_id: beam_id of the beam.
        It can be int or List[int]. If int, ndarray is return, If List[int], list[ndarray] is returned
        :param finest_grid:Default to False. If set to true, it will return 2d grid in finest resolution
        :return: ndarray/List[ndarray]
        """

        if beam_id is not None:
            ind = []
            if isinstance(beam_id, int) or isinstance(beam_id, str):
                ind = [i for i in range(len(self.beams_dict['ID'])) if
                       self.beams_dict['ID'][i] == beam_id]
                if not ind:
                    raise ValueError("invalid beam id {}".format(beam_id))
            elif isinstance(beam_id, list):
                for idx in beam_id:
                    try:
                        ind_1 = [i for i in range(len(self.beams_dict['ID'])) if
                                 self.beams_dict['ID'][i] == idx][0]
                        ind.append(ind_1)
                    except:
                        raise ValueError("invalid beam id {}".format(idx))
        # Bug fix. in case ind is int make it as list
        if isinstance(ind, int) or isinstance(ind, str):
            ind = [ind]
        beam_orig = []
        for b in ind:
            beam_map = self.beams_dict['beamlet_idx_2d_finest_grid'][b]
            if not finest_grid:
                beam_map = self._create_2d_orig_grid(beam_map)
            beam_orig.append(beam_map)
        if len(beam_orig) == 1:
            beam_orig = beam_orig[0]  # return ndarray in case if it is not list

        return beam_orig

    @staticmethod
    def get_finest_beamlet_width() -> float:
        """

        :return: beamlet width in the original beam
        """

        return 2.5

    @staticmethod
    def get_finest_beamlet_height() -> float:
        """

        :return: beamlet height in the original beam
        """

        return 2.5

    def get_beamlet_width(self) -> float:
        """

        :return: beamlet width in the original beam
        """
        beamlets = self.beams_dict['beamlets'][0]
        return np.squeeze(beamlets['width_mm'][0])[0]

    def get_beamlet_height(self) -> float:
        """

        :return: beamlet height in the original beam
        """
        beamlets = self.beams_dict['beamlets'][0]
        return np.squeeze(beamlets['height_mm'][0])[0]

    def get_all_beam_ids(self) -> List[int]:
        return self.beams_dict['ID']

    def create_beamlet_idx_2d_finest_grid(self, beam_id: int) -> np.ndarray:
        """
        Create 2d grid for the beamlets where each element is 2.5mm*2.5mm for the given beam id from x and y coordinates of beamlets.

        :param beam_id: beam_id for the beam
        :return: 2d grid of beamlets for the beam

        """
        ind = self.beams_dict['ID'].index(beam_id)
        beamlets = self.beams_dict['beamlets'][ind]
        x_positions = beamlets['position_x_mm'][0] - beamlets['width_mm'][0]/2  # x position is center of beamlet. Get left corner
        y_positions = beamlets['position_y_mm'][0] + beamlets['height_mm'][0]/2  # y position is center of beamlet. Get top corner
        right_ind = np.argmax(x_positions)
        bottom_ind = np.argmin(y_positions)
        w_all = np.column_stack((x_positions, y_positions))  # top left corners of all beamlets
        x_coord = np.arange(np.min(x_positions), np.max(x_positions) + beamlets['width_mm'][0][right_ind].item(), 2.5)
        y_coord = np.arange(np.max(y_positions), np.min(y_positions) - beamlets['height_mm'][0][bottom_ind].item(), -2.5)

        # create mesh grid in 2.5 mm resolution
        XX, YY = np.meshgrid(x_coord, y_coord)
        beamlet_idx_2d_finest_grid = np.ones_like(XX, dtype=int)
        beamlet_idx_2d_finest_grid = beamlet_idx_2d_finest_grid * int(-1)  # make all elements to -1

        for row in range(XX.shape[0]):
            for col in range(XX.shape[1]):
                ind = np.where((w_all[:, 0] == XX[row, col]) & (w_all[:, 1] == YY[row, col]))  # find the position in matrix where we find the beamlet
                if np.size(ind) > 0:
                    ind = ind[0][0]
                    num_width = int(beamlets['width_mm'][0][ind].item() /2.5)
                    num_height = int(beamlets['height_mm'][0][ind].item() / 2.5)
                    beamlet_idx_2d_finest_grid[row:row+num_height, col:col+num_width] = ind
        return beamlet_idx_2d_finest_grid

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
