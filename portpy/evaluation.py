from scipy import interpolate
import numpy as np
from portpy.beam import Beams
from portpy.structures import Structures
from portpy.clinical_criteria import ClinicalCriteria


class Evaluation:

    # def __init__(self):
    #     self.structures = None
    def __init__(self, beams: Beams, structures: Structures, clinical_criteria: ClinicalCriteria):
        self._structures = structures
        self._clinical_criteria = clinical_criteria
        self._beams = beams

    def get_dose(self, dose=None, struct=None, volume_per=None, weight_flag=False):
        x, y = self.get_dvh(dose, struct, weight_flag=weight_flag)
        f = interpolate.interp1d(100 * y, x)

        return f(volume_per)

    def get_volume(self, dose=None, struct=None, dose_value=None, weight_flag=False):
        x, y = self.get_dvh(dose=dose, struct=struct, weight_flag=weight_flag)
        x1, indices = np.unique(x, return_index=True)
        y1 = y[indices]
        f = interpolate.interp1d(x1, 100 * y1)

        return f(dose_value)

    def get_dvh(self, dose=None, struct=None, weight_flag=False):
        vox = self._structures.get_voxels_idx(struct)
        org_sort_dose = np.sort(dose[vox])
        sort_ind = np.argsort(dose[vox])
        org_sort_dose = np.append(org_sort_dose, org_sort_dose[-1] + 0.01)
        x = org_sort_dose
        if weight_flag:
            # org_points_sort_spacing = self._structures.opt_voxels_dict['dose_voxel_resolution_XYZ_mm']
            # org_points_sort_volume = org_points_sort_spacing[:, 0] * org_points_sort_spacing[:,
            #                                                          1] * org_points_sort_spacing[:, 2]
            # sum_weight = np.sum(org_points_sort_volume)
            org_weights = self._structures.get_voxels_weights(struct)
            org_sort_weights = org_weights[sort_ind]
            sum_weight = np.sum(org_sort_weights)
            y = [1]
            for j in range(len(org_sort_weights)):
                y.append(y[-1] - org_sort_weights[j] / sum_weight)
        else:
            y = np.ones(len(vox) + 1) - np.arange(0, len(vox) + 1) / len(vox)
        y[-1] = 0
        y = np.array(y)
        return x, y