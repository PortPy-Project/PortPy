import numpy as np
import scipy
import matplotlib.pyplot as plt


class Structures:
    """
    A class representing beams.
    """

    # def __init__(self, ID, gantry_angle=None, collimator_angle=None, iso_center=None, beamlet_map_2d=None,
    #              BEV_structure_mask=None, beamlet_width_mm=None, beamlet_height_mm=None, beam_modality=None, MLC_type=None):
    def __init__(self, structures, opt_voxels):
        self.structures_dict = structures
        self.opt_voxels_dict = opt_voxels
        self.preprocess_structures()

    def get_voxels_idx(self, structure_name):
        ind = self.structures_dict['name'].index(structure_name)
        vox_ind = self.structures_dict['voxel_idx'][ind]
        # vox_ind = np.where(self.opt_voxels_dict['voxel_structure_map'][0][:, ind] == 1)[0]
        return vox_ind

    def get_volume_cc(self, structure_name):
        ind = self.structures_dict['name'].index(structure_name)
        return self.structures_dict['volume_cc'][ind]

    def preprocess_structures(self):
        self.structures_dict['voxel_idx'] = [None]*len(self.structures_dict['name'])
        for i in range(len(self.structures_dict['name'])):
            self.structures_dict['voxel_idx'][i] = np.where(self.opt_voxels_dict['voxel_structure_map'][0][:, i] == 1)[0]

