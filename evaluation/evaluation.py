from scipy import interpolate
import numpy as np


class Evaluation:

    def __init__(self):
        self.structures = None

    def get_dose(self, dose, organ, volume_per, weight_flag=True):
        x, y = self.get_dvh(dose, organ, weight_flag=weight_flag)
        f = interpolate.interp1d(100 * y, x)

        return f(volume_per)

    def get_volume(self, dose, organ, dose_value, weight_flag=True):
        x, y = self.get_dvh(dose, organ, weight_flag=weight_flag)
        x1, indices = np.unique(x, return_index=True)
        y1 = y[indices]
        f = interpolate.interp1d(x1, 100 * y1)

        return f(dose_value)

    def get_dvh(self, dose, organ, weight_flag=True):
        vox = self.structures.get_voxels_idx(organ)
        org_sort_dose = np.sort(dose[vox - 1])
        sort_ind = np.argsort(dose[vox - 1])
        org_sort_dose = np.append(org_sort_dose, org_sort_dose[-1] + 0.01)
        x = org_sort_dose
        if weight_flag:
            org_points_sort_spacing = self.structures.opt_voxels_dict['voxel_resolution_XYZ_mm'][0][sort_ind]
            org_points_sort_volume = org_points_sort_spacing[:, 0] * org_points_sort_spacing[:,
                                                                     1] * org_points_sort_spacing[:, 2]
            sum_weight = np.sum(org_points_sort_volume)
            y = [1]
            for j in range(len(org_points_sort_volume)):
                y.append(y[-1] - org_points_sort_volume[j] / sum_weight)
        else:
            y = np.ones(len(vox) + 1) - np.arange(0, len(vox) + 1) / len(vox)
        y[-1] = 0
        y = np.array(y)
        return x, y