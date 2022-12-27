import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# import cv2
from shapely.geometry import LinearRing, Polygon, Point
# from skimage.draw import polygon as poly
from portpy.influence_matrix import InfluenceMatrix


class Beams:
    """
    A class representing beams.
    """

    def __init__(self, beams, opt_beamlets_PTV_margin_mm=None):

        self.opt_sol = None
        self.beams_dict = beams
        if opt_beamlets_PTV_margin_mm is None:
            opt_beamlets_PTV_margin_mm = 3
        self.opt_beamlets_PTV_margin_mm = opt_beamlets_PTV_margin_mm
        self.preprocess_beams()

    def get_beamlet_idx_2dgrid(self, beam_id=None, orig_res=True):
        # ind = np.where(np.array(self.beams_dict['ID']) == beam_id)
        ind = self.beams_dict['ID'].index(beam_id)
        if orig_res:
            return self.beams_dict['beamlet_idx_2dgrid'][ind]

    def plot_beamlet_idx_2dgrid(self, beam_id=None):
        plt.matshow(self.get_beamlet_idx_2dgrid(beam_id=beam_id))

    def get_beam_angle(self, beam_id=None):
        ind = self.beams_dict['ID'].index(beam_id)
        return self.beams_dict['gantry_angle'][ind]

    def get_collimator_angle(self, beam_id=None):
        ind = self.beams_dict['ID'].index(beam_id)
        return self.beams_dict['collimator_angle'][ind]

    def get_optimization_beamlets(self, beam_ids=None):
        if beam_ids is None:
            beam_ids = self.beams_dict['ID']
        all_beamlets = np.array([])
        for i, beam_id in enumerate(beam_ids):
            ind = self.beams_dict['ID'].index(beam_id)
            start_beamlet = self.beams_dict['start_beamlet'][ind]
            end_beamlet = self.beams_dict['end_beamlet'][ind]
            all_beamlets = np.append(all_beamlets,np.arange(start_beamlet, end_beamlet+1))
        return all_beamlets

    # def add_beam(self, beam_id=None, *args, **kwargs):
    #     old_ids = self.beams_dict['ID']
    #     new_ids = old_ids.append(beam_id)
    #     plan = Plan(beam_indices=new_ids, *args, **kwargs)

    # since instance method can modify the class variables
    @staticmethod
    def sort_beamlets(b_map):
        c = b_map[b_map >= 0]
        ind = np.arange(0, len(c))
        c_sort = np.sort(c)
        matrix_ind = [np.where(b_map == c_i) for c_i in c_sort]
        map_copy = b_map.copy()
        for i in range(len(ind)):
            map_copy[matrix_ind[i]] = ind[i]
        return map_copy

    def preprocess_beams(self, structure='PTV'):
        opt_beamlets_ids_all_beams = []
        for i, beam_id in enumerate(self.beams_dict['ID']):
            ind = self.beams_dict['ID'].index(beam_id)
            beam_2d_grid = self.create_beamlet_idx_2d_grid(beam_id=beam_id)
            self.beams_dict.setdefault('beamlet_idx_2dgrid', []).append(beam_2d_grid)

    @staticmethod
    def get_original_map(beam_map):
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

    def get_fluence_map(self, beam_id, optimal_intensity=None, scenario=0):
        # Generate the beamlet maps from w
        if beam_id is None:
            beam_ids = self.beams_dict['ID'][0]
        if optimal_intensity is None:
            optimal_intensity = self.opt_sol[scenario]['optimal_intensity']
        # wMaps = []
        # for b, beam_id in enumerate(beam_ids):
        ind = self.beams_dict['ID'].index(beam_id)
        maps = self.beams_dict['beamlet_idx_2dgrid'][ind]
        numRows = np.size(maps, 0)
        numCols = np.size(maps, 1)
        # wMaps.append(np.zeros((numRows, numCols)))
        wMaps = np.zeros((numRows, numCols))
        for r in range(numRows):
            for c in range(numCols):
                if maps[r, c] >= 0:
                    curr = maps[r, c]
                    # wMaps[b][r, c] = optimal_intensity[curr]
                    wMaps[r, c] = optimal_intensity[curr]

        return wMaps

    def create_beamlet_idx_2d_grid(self, beam_id=None):
        ind = self.beams_dict['ID'].index(beam_id)
        beamlets = self.beams_dict['beamlets'][ind]
        x_positions = beamlets['position_x_mm'][0] - beamlets['width_mm'][0]/2
        y_positions = beamlets['position_y_mm'][0] + beamlets['height_mm'][0]/2
        # top_left_ind = np.where((x_positions == np.min(x_positions)) & (y_positions == np.max(y_positions)))[0][0]
        # left_most_x = np.min(x_positions) - beamlets['width'][0][top_left_ind]/2
        # top_most_y = np.max(y_positions) + beamlets['height'][0][top_left_ind]/2
        # bottom_right_ind = np.where((x_positions == np.max(x_positions)) & (y_positions == np.min(y_positions)))[0][0]
        # right_x = np.max(x_positions) - beamlets['width'][0][bottom_right_ind] / 2
        # bottom_y = np.min(y_positions) + beamlets['height'][0][bottom_right_ind] / 2
        right_ind = np.argmax(x_positions)
        bottom_ind = np.argmin(y_positions)
        w_all = np.column_stack((x_positions, y_positions))  # top left corners of all beamlets
        x_coord = np.arange(np.min(x_positions), np.max(x_positions) + beamlets['width_mm'][0][right_ind], 2.5)
        y_coord = np.arange(np.max(y_positions), np.min(y_positions) - beamlets['height_mm'][0][bottom_ind], -2.5)
        XX, YY = np.meshgrid(x_coord, y_coord)
        beamlet_idx_2d_grid = np.ones_like(XX, dtype=int)
        beamlet_idx_2d_grid = beamlet_idx_2d_grid * int(-1)
        for row in range(XX.shape[0]):
            for col in range(XX.shape[1]):
                ind = np.where((w_all[:, 0] == XX[row, col]) & (w_all[:, 1] == YY[row, col]))
                if np.size(ind) > 0:
                    ind = ind[0][0]
                    num_width = int(beamlets['width_mm'][0][ind]/2.5)
                    num_height = int(beamlets['height_mm'][0][ind] / 2.5)
                    beamlet_idx_2d_grid[row:row+num_height, col:col+num_width] = ind
        return beamlet_idx_2d_grid
