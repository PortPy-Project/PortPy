from scipy import interpolate
import numpy as np


class Evaluation:

    @staticmethod
    def get_dose(sol: dict, struct: str, volume_per: float, dose_1d: np.ndarray = None, weight_flag: bool = True) -> float:
        """
        Get dose_1d at volume percentage

        :param sol: solution dictionary
        :param dose_1d: dose_1d in 1d
        :param struct: structure name for which to get the dose_1d
        :param volume_per: query the dose_1d at percentage volume
        :param weight_flag: for non uniform voxels weight flag always True
        :return: dose_1d at volume_percentage

        :Example:

        >>> Evaluation.get_dose(sol=sol, struct='PTV', volume_per=90)

        """
        x, y = Evaluation.get_dvh(sol, dose_1d=dose_1d, struct=struct, weight_flag=weight_flag)
        f = interpolate.interp1d(100 * y, x)

        return f(volume_per)

    @staticmethod
    def get_volume(sol: dict, struct: str, dose_value_gy: float, dose_1d: np.ndarray = None, weight_flag: bool = True) -> float:
        """
        Get volume at dose_1d value in Gy

        :param sol: solution dictionary
        :param dose_1d: dose_1d in 1d
        :param struct: structure name for which to get the dose_1d
        :param dose_value_gy: query the volume at dose_value
        :param weight_flag: for non uniform voxels weight flag always True
        :return: dose_1d at volume_percentage

        :Example:

        >>> Evaluation.get_volume(sol=sol, struct='PTV', dose_value_gy=60)

        """
        x, y = Evaluation.get_dvh(sol, dose_1d=dose_1d, struct=struct, weight_flag=weight_flag)
        x1, indices = np.unique(x, return_index=True)
        y1 = y[indices]
        f = interpolate.interp1d(x1, 100 * y1)
        if dose_value_gy > max(x1):
            print('Warning: dose_1d value {} is greater than max dose_1d for {}'.format(dose_value_gy, struct))
            return 0
        else:
            return f(dose_value_gy)

    @staticmethod
    def get_dvh(sol: dict, struct: str, dose_1d: np.ndarray = None, weight_flag: bool = True):
        """
        Get dvh for the structure

        :param sol: optimal solution dictionary
        :param dose_1d: dose_1d which is not in solution dictionary
        :param struct: structure name
        :param weight_flag: for non uniform voxels weight flag always True
        :return: x, y --> dvh for the structure

        :Example:

        >>> Evaluation.get_dvh(sol=sol, struct='PTV')

        """
        inf_matrix = sol['inf_matrix']
        vox = inf_matrix.get_opt_voxels_idx(struct)
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        org_sort_dose = np.sort(dose_1d[vox])
        sort_ind = np.argsort(dose_1d[vox])
        org_sort_dose = np.append(org_sort_dose, org_sort_dose[-1] + 0.01)
        x = org_sort_dose
        if weight_flag:
            # org_points_sort_spacing = my_plan._structures.opt_voxels_dict['dose_voxel_resolution_XYZ_mm']
            # org_points_sort_volume = org_points_sort_spacing[:, 0] * org_points_sort_spacing[:,
            #                                                          1] * org_points_sort_spacing[:, 2]
            # sum_weight = np.sum(org_points_sort_volume)
            org_weights = inf_matrix.get_opt_voxels_size(struct)
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

    @staticmethod
    def get_max_dose(sol: dict, struct: str, dose_1d=None) -> float:
        """
        Get maximum dose_1d for the structure

        :param sol: optimal solution dictionary
        :param dose_1d: dose_1d which is not in solution dictionary
        :param struct: structure name

        :return: maximum dose_1d for the structure
        """
        inf_matrix = sol['inf_matrix']
        vox = inf_matrix.get_opt_voxels_idx(struct)
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        return np.max(dose_1d[vox])

    @staticmethod
    def get_mean_dose(sol: dict, struct: str, dose_1d=None) -> np.ndarray:
        """
                Get mean dose_1d for the structure

                :param sol: optimal solution dictionary
                :param dose_1d: dose_1d which is not in solution dictionary
                :param struct: structure name

                :return: mean dose_1d for the structure
                """
        inf_matrix = sol['inf_matrix']
        vox = inf_matrix.get_opt_voxels_idx(struct)
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        return np.mean(dose_1d[vox])


