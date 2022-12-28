import numpy as np
from shapely.geometry import LinearRing, Polygon, Point
from scipy import sparse
from typing import Optional
from copy import deepcopy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import itertools


class InfluenceMatrix:
    """
    class of influence matrix
    """

    def __init__(self, plan_obj,
                 opt_beamlets_PTV_margin_mm: Optional['float'] = None,
                 beamlet_width=2.5, beamlet_height=2.5,
                 structure='PTV'):
        """

        :param plan_obj: plan object
        :param opt_beamlets_PTV_margin_mm:
        :param beamlet_width: Default beamlet width 2.5
        :param beamlet_height: Default beamlet height 2.2
        :param structure: target structure for creating BEV beamlets
        """
        # create deepcopy of the object or else it will modify the my_plan object
        self._beamlets_in_inf_matrix = []
        self.beamlet_width = beamlet_width
        self.beamlet_height = beamlet_height

        if hasattr(plan_obj.structures, 'opt_voxels_dict'):
            self.opt_voxels_dict = deepcopy(plan_obj.structures.opt_voxels_dict)
            del plan_obj.structures.opt_voxels_dict
        else:
            self.opt_voxels_dict = deepcopy(plan_obj.inf_matrix.opt_voxels_dict)

        if 'beamlets' in plan_obj.beams.beams_dict:
            self.beamlets_dict = deepcopy(plan_obj.beams.beams_dict['beamlets'])
        else:
            self.beamlets_dict = deepcopy(plan_obj.inf_matrix.beamlets_dict['beamlets'])

        for i in range(len(plan_obj.beams.beams_dict['ID'])):
            self.beamlets_dict[i]['beam_id'] = plan_obj.beams.beams_dict['ID'][i]

        if opt_beamlets_PTV_margin_mm is None:
            self.opt_beamlets_PTV_margin_mm = plan_obj.opt_beamlets_PTV_margin_mm
        else:
            self.opt_beamlets_PTV_margin_mm = opt_beamlets_PTV_margin_mm
        print('Creating BEV')
        self.preprocess_beams(plan_obj, structure=structure)
        self.A = self.get_influence_matrix(plan_obj)
        self.dose_3d = None

        print('Done')

    def get_voxel_info(self, row_number):
        pass

    def get_beamlet_info(self, col_number):
        pass

    def dose_1d_to_3d(self, sol: dict = None, dose_1d=None):
        """

        :param sol: solution from optimization
        :param dose_1d: dose in 1d. Optional
        :return: dose in 3d

        Get dose in 3d array. same resolution as CT
        """
        dose_vox_map = self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        # dose_1d = self.opt_voxels_dict['dose_1d']
        if dose_1d is None:
            if 'dose_1d' not in sol:
                dose_1d = sol['inf_matrix'].A * sol['optimal_intensity']  # multiply it with num fractions
            else:
                dose_1d = sol['dose_1d']
        dose_3d = np.zeros_like(dose_vox_map, dtype=float)
        inds = np.unique(dose_vox_map[dose_vox_map >= 0])
        a = np.where(np.isin(dose_vox_map, inds))
        dose_3d[a] = dose_1d[dose_vox_map[a]]
        self.dose_3d = dose_3d
        return dose_3d

    def dose_3d_to_1d(self, dose_3d: np.ndarray = None):
        """

        :param dose_3d: 3d dose
        :return: dose in 1d voxel indices
        """
        dose_vox_map = self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        if dose_3d is None:
            dose_3d = self.dose_3d
        inds = np.unique(dose_vox_map[dose_vox_map >= 0])
        dose_1d = np.zeros_like(inds, dtype=float)
        a = np.where(np.isin(dose_vox_map, inds))
        dose_1d[dose_vox_map[a]] = dose_3d[a]
        return dose_1d

    def fluence_2d_to_1d(self, fluence_2d: list):
        """

        :param fluence_2d: 2d fluence as list of nd array with same length as number of beams
        :return: fluence in 1d for beamlet indices
        """
        fluence_1d = np.zeros((self.A.shape[1]))
        for ind in range(len(self.beamlets_dict)):
            maps = self.beamlets_dict[ind]['beamlet_idx_2dgrid']
            numRows = np.size(maps, 0)
            numCols = np.size(maps, 1)
            # wMaps = np.zeros((numRows, numCols))
            for r in range(numRows):
                for c in range(numCols):
                    if maps[r, c] >= 0:
                        curr = maps[r, c]
                        fluence_1d[curr] = fluence_2d[ind][r, c]
        return fluence_1d

    def fluence_1d_to_2d(self, sol: dict):
        """

        :param sol: solution from optimization
        :return: 2d fluence as a list for each beam

        """
        # if optimal_intensity is None:
        #     optimal_intensity = self.opt_sol[scenario]['optimal_intensity']
        wMaps = []
        for ind in range(len(self.beamlets_dict)):
            maps = self.beamlets_dict[ind]['beamlet_idx_2dgrid']
            numRows = np.size(maps, 0)
            numCols = np.size(maps, 1)
            wMaps.append(np.zeros((numRows, numCols)))
            # wMaps = np.zeros((numRows, numCols))
            for r in range(numRows):
                for c in range(numCols):
                    if maps[r, c] >= 0:
                        curr = maps[r, c]
                        wMaps[ind][r, c] = sol['optimal_intensity'][curr]
                        # wMaps[r, c] = optimal_intensity[curr]

        return wMaps

    def get_influence_matrix(self, plan, is_sparse=True):
        # if beam_ids is None:
        #     beam_ids = self.beamlets_dict['ID']
        # beam_ids_list = beam_ids if isinstance(beam_ids, list) else [beam_ids]
        # for i, beam_id in enumerate(beam_ids_list):
        if self.beamlet_width > 2.5 or self.beamlet_height > 2.5:
            influenceMatrixSparse = plan.inf_matrix.A
        else:
            influenceMatrixSparse = plan.beams.beams_dict['influenceMatrixSparse']

        for ind in range(len(self.beamlets_dict)):
            # ind = self.beamlets_dict['ID'].index(beam_id)
            # beamlet_idx_2dgrid = self.get_beamlet_idx_2dgrid(beam_id=beam_id)
            # opt_beamlets = beamlet_idx_2dgrid[beamlet_idx_2dgrid >= 0]
            opt_beamlets = self.beamlets_dict[ind]['opt_beamlets_ids']
            print('parsing influence matrix for beam {}'.format(ind))
            if is_sparse:
                if ind == 0:
                    if self.beamlet_width > 2.5 or self.beamlet_height > 2.5:

                        # inf_matrix = lil_matrix((influenceMatrixSparse[ind].shape[0], len(opt_beamlets)))
                        # inf_matrix = sparse.hstack([csr_matrix(influenceMatrixSparse[ind][:, np.unique(opt_beamlets[i])].sum(axis=1)) for i in range(len(opt_beamlets))], format='csr')
                        inf_matrix = sparse.hstack(
                            [csr_matrix(influenceMatrixSparse[:, np.unique(opt_beamlets[i])].sum(axis=1)) for i in
                             range(len(opt_beamlets))], format='csr')
                        # for i in range(len(opt_beamlets)):
                        #     inf_matrix[:, i] = influenceMatrixSparse[ind][:, np.unique(opt_beamlets[i])].sum(axis=1)
                        # inf_matrix = np.vstack([np.sum(influenceMatrixSparse[ind][:, opt_beamlets[n]], 1) for n in range(len(opt_beamlets))]).T
                    else:
                        inf_matrix = influenceMatrixSparse[ind][:, opt_beamlets]
                else:
                    if self.beamlet_width > 2.5 or self.beamlet_height > 2.5:
                        inf_matrix_2 = sparse.hstack(
                            [csr_matrix(influenceMatrixSparse[:, np.unique(opt_beamlets[i])].sum(axis=1)) for i in
                             range(len(opt_beamlets))], format='csr')
                        inf_matrix = sparse.hstack(
                            [inf_matrix, inf_matrix_2], format='csr')
                    else:
                        inf_matrix = sparse.hstack(
                            [inf_matrix, influenceMatrixSparse[ind][:, opt_beamlets]], format='csr')
        # if del_org_matrix:
        if 'influenceMatrixSparse' in plan.beams.beams_dict:
            del plan.beams.beams_dict['influenceMatrixSparse']

        return inf_matrix

    def create_BEV_mask_from_contours(self, plan_obj, ind=None, structure='PTV', margin=None):
        if margin is None and structure == 'PTV':
            margin = self.opt_beamlets_PTV_margin_mm
        elif margin is None:
            margin = 0

        # get PTV contour from data
        contours = plan_obj.beams.beams_dict['BEV_structure_contour_points'][ind][structure]
        # ind = self.beamlets_dict['ID'].index(beam_id)
        # contours = self._beams_contours[ind][structure]

        # for each contour create polygon and get beamlets inside the polygon and create mask for it
        for count_num in range(len(contours)):
            polygon = []
            for j in contours[count_num]:
                polygon.append((j[0], j[1]))
            r = LinearRing(polygon)
            s = Polygon(r)
            # add margin around polygon
            if margin == 0:
                shapely_poly = s
            else:
                shapely_poly = Polygon(s.buffer(margin), [r])
            #             poly_coordinates = np.array(list(t.exterior.coords))

            # create beam map from beamlet coordinates and mask the beamlets inside shapely polygon
            beamlets = self.beamlets_dict[ind]
            x_positions = beamlets['position_x_mm'][0]
            y_positions = beamlets['position_y_mm'][0]
            x_min_max_sort = np.sort(np.unique(x_positions))
            y_max_min_sort = np.sort(np.unique(y_positions))[::-1]
            points = []
            for i in range(len(beamlets['id'][0])):
                x_coord = beamlets['position_x_mm'][0][i]
                y_coord = beamlets['position_y_mm'][0][i]
                points.append(Point(x_coord, y_coord))
            valid_points = []
            valid_points.extend(filter(shapely_poly.contains, points))
            x_and_y = [(a.x, a.y) for a in valid_points]
            XX, YY = np.meshgrid(x_min_max_sort, y_max_min_sort)
            w_all = np.column_stack((x_positions, y_positions))
            if count_num == 0:
                mask = np.zeros_like(XX, dtype=bool)
            for row in range(XX.shape[0]):
                for col in range(XX.shape[1]):
                    ind = np.where((w_all[:, 0] == XX[row, col]) & (w_all[:, 1] == YY[row, col]))[0][0]

                    # if (beamlets[ind]['position_x_mm'], beamlets[ind]['position_y_mm']) in x_and_y:
                    if (beamlets['position_x_mm'][0][ind], beamlets['position_y_mm'][0][ind]) in x_and_y:
                        # beam_map[row, col] = beamlets[ind]['id']
                        mask[row, col] = True
        return mask

    def preprocess_beams(self, plan_obj, structure='PTV', remove_corner_beamlets=False):
        opt_beamlets_ids_all_beams = []
        for ind in range(len(self.beamlets_dict)):
            # ind = self.beamlets_dict['ID'].index(beam_id)
            mask = self.create_BEV_mask_from_contours(plan_obj, ind=ind, structure=structure,
                                                      margin=self.opt_beamlets_PTV_margin_mm)
            beam_2d_grid = self.create_beamlet_idx_2d_grid(ind=ind)
            beamlets = self.beamlets_dict[ind]

            # creating beam_map of original resolution so that mask can be multiplied with it
            beam_map = self.get_orig_res_2d_grid(ind)
            beam_map = np.multiply((beam_map + int(1)), mask)  # add and subtract one to maintain 0th beamlet
            beam_map = beam_map - np.int(1)  # subtract one again to get original beamlets

            # get mask and beam_map in 2.5mm resolution
            beamlet_ind = np.unique(beam_map.flatten())
            beamlet_ind = beamlet_ind[beamlet_ind >= 0]
            a = np.where(np.isin(beam_2d_grid, beamlet_ind))
            mask_2d_grid = np.zeros_like(beam_2d_grid, dtype=bool)
            mask_2d_grid[a] = True
            beam_2d_grid = (beam_2d_grid + int(1)) * mask_2d_grid  # add and subtract 1 to retain ids
            beam_2d_grid = beam_2d_grid - int(1)
            beam_map = beam_2d_grid

            # get opt beamlets for down_sample grid as list of original inf_matrix beamlet
            if self.beamlet_width > 2.5 or self.beamlet_height > 2.5:
                down_sample_2d_grid = self.down_sample_2d_grid(ind=ind, beamlet_width=self.beamlet_width,
                                                               beamlet_height=self.beamlet_height)
                down_sample_2d_grid = (down_sample_2d_grid + int(1)) * mask_2d_grid  # add and subtract 1 to retain ids
                down_sample_2d_grid = down_sample_2d_grid - int(1)
                down_sample_beamlets, counts = np.unique(np.sort(down_sample_2d_grid[down_sample_2d_grid >= 0]),
                                                         return_counts=True)
                # remove corner beamlets in coarse resolution and updating down sample map
                if remove_corner_beamlets:
                    count_ind = np.where(counts >= (self.beamlet_width / 2.5) * (self.beamlet_height / 2.5))
                    down_sample_beamlets = down_sample_beamlets[count_ind]
                    # updating down sample grid after removing corner beamlets
                    inds_down_sample = down_sample_2d_grid == down_sample_beamlets[:, None,
                                                              None]  # keep only down sample beamlets and remove others
                    down_sample_2d_grid[~np.any(inds_down_sample, axis=0)] = -1

                a = np.where(np.isin(down_sample_2d_grid, down_sample_beamlets))
                mask_2d_grid = np.zeros_like(beam_2d_grid, dtype=bool)
                mask_2d_grid[a] = True
                # actual_beamlets = beam_2d_grid[a]
                actual_beamlets = plan_obj.inf_matrix.beamlets_dict[ind]['beamlet_idx_2dgrid'][a]
                sampled_beamlets = down_sample_2d_grid[a]
                b = [np.where(sampled_beamlets == down_sample_beamlets[i]) for i in range(len(down_sample_beamlets))]
                opt_beamlets = [actual_beamlets[i] for i in b]

                beam_map = down_sample_2d_grid
            else:
                opt_beamlets = np.unique(np.sort(beam_map[beam_map >= 0]))

            # make beamlets continuous
            std_map = self.sort_beamlets(beam_map)
            if ind == 0:
                beam_map = std_map
            else:
                beam_map = std_map + np.int(
                    np.amax(self.beamlets_dict[ind - 1]['beamlet_idx_2dgrid']) + 1) * mask_2d_grid
            standInd = np.unique(np.sort(beam_map.flatten()))
            self.beamlets_dict[ind]['beamlet_idx_2dgrid'] = beam_map
            # self.beamlets_dict.setdefault('structure_mask_2dgrid', []).append(mask_2d_grid)
            self.beamlets_dict[ind]['start_beamlet'] = standInd[1]
            self.beamlets_dict[ind]['end_beamlet'] = np.amax(self.beamlets_dict[ind]['beamlet_idx_2dgrid'])
            self.beamlets_dict[ind]['opt_beamlets_ids'] = opt_beamlets
            self._beamlets_in_inf_matrix.append(opt_beamlets)

            # update beamlet dic
            if self.beamlet_width > 2.5 or self.beamlet_height > 2.5:
                pass
            else:
                self.beamlets_dict[ind]['position_x_mm'][0] = beamlets['position_x_mm'][0][opt_beamlets]
                self.beamlets_dict[ind]['position_y_mm'][0] = beamlets['position_y_mm'][0][opt_beamlets]
                self.beamlets_dict[ind]['width_mm'][0] = beamlets['width_mm'][0][opt_beamlets]
                self.beamlets_dict[ind]['height_mm'][0] = beamlets['height_mm'][0][opt_beamlets]
                self.beamlets_dict[ind]['MLC_leaf_idx'][0] = beamlets['MLC_leaf_idx'][0][opt_beamlets]
            del self.beamlets_dict[ind]['id']
            # self.beams_dict.setdefault('opt_beamlets_ids', []).append(standInd[1:])

    def create_beamlet_idx_2d_grid(self, ind=None):
        # ind = self.beamlets_dict['ID'].index(beam_id)
        beamlets = self.beamlets_dict[ind]
        x_positions = beamlets['position_x_mm'][0] - beamlets['width_mm'][0] / 2
        y_positions = beamlets['position_y_mm'][0] + beamlets['height_mm'][0] / 2
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
                    num_width = int(beamlets['width_mm'][0][ind] / 2.5)
                    num_height = int(beamlets['height_mm'][0][ind] / 2.5)
                    beamlet_idx_2d_grid[row:row + num_height, col:col + num_width] = ind

        return beamlet_idx_2d_grid

    def down_sample_2d_grid(self, ind=None, beamlet_width=5.0, beamlet_height=5.0):

        # ind = self.beamlets_dict['ID'].index(beam_id)
        beamlets = self.beamlets_dict[ind]
        x_positions = beamlets['position_x_mm'][0] - beamlets['width_mm'][0] / 2
        y_positions = beamlets['position_y_mm'][0] + beamlets['height_mm'][0] / 2
        right_ind = np.argmax(x_positions)
        bottom_ind = np.argmin(y_positions)
        # w_all = np.column_stack((x_positions, y_positions))  # top left corners of all beamlets
        x_coord = np.arange(np.min(x_positions), np.max(x_positions) + beamlets['width_mm'][0][right_ind], 2.5)
        y_coord = np.arange(np.max(y_positions), np.min(y_positions) - beamlets['height_mm'][0][bottom_ind], -2.5)
        XX, YY = np.meshgrid(x_coord, y_coord)
        beamlet_resample_2d_grid = np.ones_like(XX, dtype=int)
        beamlet_resample_2d_grid = beamlet_resample_2d_grid * int(-1)
        row_col_covered = []
        ind = 0
        for row in range(XX.shape[0]):
            for col in range(XX.shape[1]):
                # ind = np.where((w_all[:, 0] == XX[row, col]) & (w_all[:, 1] == YY[row, col]))
                # if np.size(ind) > 0:
                #     ind = ind[0][0]
                if [row, col] not in row_col_covered:
                    num_width = int(beamlet_width / 2.5)
                    num_height = int(beamlet_height / 2.5)
                    beamlet_resample_2d_grid[row:row + num_height, col:col + num_width] = ind
                    ind = ind + 1
                    rows = np.arange(row, row + num_height)
                    cols = np.arange(col, col + num_width)
                    for r in itertools.product(rows, cols):
                        row_col_covered.append([r[0], r[1]])
        return beamlet_resample_2d_grid

    def get_orig_res_2d_grid(self, ind):
        beamlets = self.beamlets_dict[ind]
        x_positions = beamlets['position_x_mm'][0]
        y_positions = beamlets['position_y_mm'][0]
        x_min_max_sort = np.sort(np.unique(x_positions))
        y_max_min_sort = np.sort(np.unique(y_positions))[::-1]
        XX, YY = np.meshgrid(x_min_max_sort, y_max_min_sort)
        w_all = np.column_stack((x_positions, y_positions))
        beam_map = np.zeros_like(XX, dtype=int)
        for row in range(XX.shape[0]):
            for col in range(XX.shape[1]):
                b_ind = np.where((w_all[:, 0] == XX[row, col]) & (w_all[:, 1] == YY[row, col]))[0][0]
                # beam_map[row, col] = beamlets[ind]['id']+np.int(1)# adding one so that 0th beamlet is retained
                beam_map[row, col] = b_ind  # + np.int(1)  # adding one so that 0th beamlet is retained
        # beam_map = np.multiply(beam_map, mask)
        # beam_map = beam_map - np.int(1)  # subtract one again to get original beamlets
        return beam_map

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

    # for voxels idx methods
    def set_opt_voxel_idx(self, plan_obj, structure_name: str):
        """

        :param plan_obj: object of class Plan
        :param structure_name: structure name
        :return: set the voxel idx in opt_voxels_dict for the structure
        """
        ind = plan_obj.structures.structures_dict['name'].index(structure_name)
        vox_3d = plan_obj.structures.structures_dict['structure_mask_3d'][ind] * \
                 self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        # self.structures_dict['voxel_idx'][i] = np.unique(vox_3d[vox_3d > 0])
        vox, counts = np.unique(vox_3d[vox_3d > 0], return_counts=True)
        self.opt_voxels_dict['voxel_idx'].append(vox)
        self.opt_voxels_dict['voxel_size'].append(counts / np.max(counts))  # calculate weight for each voxel
        self.opt_voxels_dict['name'].append(structure_name)

    def get_opt_voxels_idx(self, structure_name: str):
        """
        Get voxel index for structure
        :param structure_name: name of the structure in plan
        :return: voxel indexes for the structure
        """
        ind = self.opt_voxels_dict['name'].index(structure_name)
        vox_ind = self.opt_voxels_dict['voxel_idx'][ind]
        # vox_ind = np.where(self.opt_voxels_dict['voxel_structure_map'][0][:, ind] == 1)[0]
        return vox_ind

    def get_opt_voxels_size(self, structure_name: str):
        """
         :param structure_name: name of the structure in plan
         :return: voxel size for the structure
         """
        ind = self.opt_voxels_dict['name'].index(structure_name)
        vox_weights = self.opt_voxels_dict['voxel_size'][ind]
        # vox_ind = np.where(self.opt_voxels_dict['voxel_structure_map'][0][:, ind] == 1)[0]
        return vox_weights

    def plot_fluence_2d(self, beam_id=None, optimal_fluence_2d=None):
        # Generate the beamlet maps from w
        # beam_id = beam_id if isinstance(beam_id, list) else [beam_id]
        # fluence_map_2d = self.get_fluence_map(beam_id=beam_id)
        # ind = self.beamlets_dict['beam_id'].index(beam_id)
        ind = [i for i in range(len(self.beamlets_dict)) if self.beamlets_dict[i]['beam_id'] == beam_id][0]
        plt.matshow(optimal_fluence_2d[ind])
        plt.xlabel('x-axis (beamlets column)')
        plt.ylabel('y-axis (beamlets row)')

    def plot_fluence_3d(self, beam_id=None, optimal_fluence_2d=None):
        # beam_id = beam_id if isinstance(beam_id, list) else [beam_id]
        # fluence_map_2d = self.get_fluence_map(beam_id=beam_id)
        # ind = self.beamlets_dict['beam_id'].index(beam_id)
        ind = [i for i in range(len(self.beamlets_dict)) if self.beamlets_dict[i]['beam_id'] == beam_id][0]
        (fig, ax, surf) = InfluenceMatrix.surface_plot(optimal_fluence_2d[ind], cmap='viridis', edgecolor='black')
        ax.set_zlabel('Fluence Intensity')
        ax.set_xlabel('x-axis (beamlets column)')
        ax.set_ylabel('y-axis (beamlets row)')
        fig.colorbar(surf)
        plt.show()

    @staticmethod
    def surface_plot(matrix, **kwargs):
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, np.transpose(matrix), **kwargs)
        return fig, ax, surf
