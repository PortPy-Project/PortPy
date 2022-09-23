import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import LinearRing, Polygon
from skimage.draw import polygon as poly

class Beam:
    """
    A class representing each beam.
    """

    # def __init__(self, ID, gantry_angle=None, collimator_angle=None, iso_center=None, beamlet_map_2d=None,
    #              BEV_structure_mask=None, beamlet_width_mm=None, beamlet_height_mm=None, beam_modality=None, MLC_type=None):
    def __init__(self, data):

        self.ID = data['ID']
        self.gantry_angle = data['gantry_angle']
        self.collimator_angle = data['collimator_angle']
        self.iso_center = data['iso_center']
        self.jaw_position = data['jaw_position']
        beamlet_opt = np.multiply(data['BEV_2d_beamlet_idx'], data['BEV_2d_structure_mask']['PTV'])
        self.beamlet_opt_ID = beamlet_opt[np.nonzero(beamlet_opt)]
        self.beamlet_opt = [data['beamlets'][i] for i in self.beamlet_opt_ID]
        self.BEV_2d_beamlet_idx = data['BEV_2d_beamlet_idx']
        self.BEV_2d_structure_mask = data['BEV_2d_structure_mask']
        self.beam_modality = data['beam_modality']
        self.MLC_type = data['MLC_type']
        self.beam_modality = data['beam_modality']

    def plot_BEV_2d_structure_mask(self, organ=None):
        plt.matshow(self.BEV_2d_structure_mask[organ])


class Beams:
    """
    A class representing beams.
    """

    # def __init__(self, ID, gantry_angle=None, collimator_angle=None, iso_center=None, beamlet_map_2d=None,
    #              BEV_structure_mask=None, beamlet_width_mm=None, beamlet_height_mm=None, beam_modality=None, MLC_type=None):
    def __init__(self, beams, opt_beamlets_PTV_margin_mm=None):
        self.beams_dict = beams
        if opt_beamlets_PTV_margin_mm is None:
            opt_beamlets_PTV_margin_mm = 3
        self.opt_beamlets_PTV_margin_mm = opt_beamlets_PTV_margin_mm

    def plot_BEV_2d_structure_mask(self, beam_id=None, organ=None):
        plt.matshow(self.get_BEV_2d_structure_mask(beam_id, organ))

    def get_BEV_2d_structure_mask(self, beam_id=None, organ=None, margin=0):
        # ind = np.where(np.array(self.beams_dict['ID']) == beam_id)
        ind = self.beams_dict['ID'].index(beam_id)
        if margin == 0:
            a_mask = self.beams_dict['BEV_2d_structure_mask'][ind][organ]
        else:
            mask = self.beams_dict['BEV_2d_structure_mask'][ind][organ]
            mask = mask.astype('uint8')
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for count_num in range(len(contours)):
                polygon = []
                for i in contours[count_num]:
                    polygon.append((i[0][0], i[0][1]))
                r = LinearRing(polygon)
                s = Polygon(r)
                t = Polygon(s.buffer(margin / 2.5).exterior, [r])

                if count_num == 0:
                    a_mask = np.zeros(shape=mask.shape[0:2])  # original
                poly_coordinates = np.array(list(t.exterior.coords))
                rr, cc = poly(poly_coordinates[:, 0], poly_coordinates[:, 1], (mask.shape[1], mask.shape[0]))
                a_mask[cc, rr] = 1
        return a_mask

    def get_2d_beamlet_idx(self, beam_id=None, organ='PTV'):

        # ind = np.where(np.array(self.beams_dict['ID']) == beam_id)
        ind = self.beams_dict['ID'].index(beam_id)
        mask = self.get_BEV_2d_structure_mask(beam_id=beam_id, organ=organ, margin=self.opt_beamlets_PTV_margin_mm)
        idx_2d = np.multiply(self.beams_dict['BEV_2d_beamlet_idx'][ind] + 1, mask)
        beam_map = np.int32(idx_2d) - np.int32(1)
        return beam_map

    def get_original_map(self, beam_map):
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






