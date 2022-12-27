import numpy as np
import scipy
import matplotlib.pyplot as plt
# import cv2
from copy import deepcopy


class Structures:
    """
    A class representing structures.
    """

    def __init__(self, structures, opt_voxels):
        # super().__init__()
        self.structures_dict = structures
        self.opt_voxels_dict = opt_voxels
        self.opt_voxels_dict['name'] = structures['name']
        self.ct_voxel_resolution_xyz_mm = deepcopy(self.opt_voxels_dict['ct_voxel_resolution_xyz_mm'])
        self.preprocess_structures()

    def get_volume_cc(self, structure_name: str):
        """
         :param structure_name: name of the structure in plan
         :return: volume of the structure
         """
        ind = self.structures_dict['name'].index(structure_name)
        return self.structures_dict['volume_cc'][ind]

    def preprocess_structures(self, down_sample=False):
        self.opt_voxels_dict['voxel_idx'] = [None] * len(self.structures_dict['name'])
        self.opt_voxels_dict['voxel_size'] = [None] * len(self.structures_dict['name'])
        for i in range(len(self.structures_dict['name'])):
            if down_sample:
                vox_3d = self.structures_dict['structure_mask_3d'][i] * self.opt_voxels_dict['ct_dose_map_down_sampled']
            else:
                vox_3d = self.structures_dict['structure_mask_3d'][i] * self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
            # self.structures_dict['voxel_idx'][i] = np.unique(vox_3d[vox_3d > 0])
            vox, counts = np.unique(vox_3d[vox_3d > 0], return_counts=True)
            self.opt_voxels_dict['voxel_idx'][i] = vox
            self.opt_voxels_dict['voxel_size'][i] = counts / np.max(counts)  # calculate weight for each voxel

    def create_structure(self, new_structure: str, mask_3d: np.ndarray):
        """
        :param new_structure: name of the new structure
        :param mask_3d: 3d mask for the structure
        :return: create new_structure
        """
        for key in self.structures_dict.keys():
            if key == 'name':
                self.structures_dict['name'].append(new_structure)
            elif key == 'structure_mask_3d':
                self.structures_dict['structure_mask_3d'].append(mask_3d)
            # elif key == 'voxel_idx':
            #     dose_vox_3d = self.structures_dict['structure_mask_3d'][-1] * \
            #                   self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
            #    self.structures_dict['voxel_idx'].append(np.unique(dose_vox_3d[dose_vox_3d > 0]))
            # vox, counts = np.unique(dose_vox_3d[dose_vox_3d > 0], return_counts=True)
            # self.opt_voxels_dict['voxel_idx'].append(vox)
            # self.opt_voxels_dict['voxel_size'].append(counts / np.max(counts))
            # elif key == 'voxel_size':
            #     continue
            else:
                self.structures_dict[key].append(None)

    def modify_structure(self, structure: str, mask_3d: np.ndarray):
        """
        :param structure: name of the structure to be modified
        :param mask_3d: 3d mask for the structure
        :return: modify structure
        """
        ind = self.structures_dict['name'].index(structure)
        for key in self.structures_dict.keys():
            if key == 'name':
                self.structures_dict['name'][ind] = structure
            elif key == 'structure_mask_3d':
                self.structures_dict['structure_mask_3d'][ind] = mask_3d
            # elif key == 'voxel_idx':
            #     dose_vox_3d = self.structures_dict['structure_mask_3d'][ind] * \
            #                   self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
            #    # self.structures_dict['voxel_idx'][ind] = np.unique(dose_vox_3d[dose_vox_3d > 0])
            #     vox, counts = np.unique(dose_vox_3d[dose_vox_3d > 0], return_counts=True)
            #     self.opt_voxels_dict['voxel_idx'][ind] = vox
            #     self.opt_voxels_dict['voxel_size'][ind] = counts / np.max(counts)
            # elif key == 'voxel_size':
            #     continue
            else:
                self.structures_dict[key][ind] = None

    def delete_structure(self, structure: str):
        """
        :param structure: structure to be removed
        :return:
        """
        ind = self.structures_dict['name'].index(structure)
        for key in self.structures_dict.keys():
            del self.structures_dict[key][ind]

    # def add_structure(self, new_structure=None):
    #     self.structures_dict['name'].append(new_structure)
    #     self.structures_dict['structure_mask_3d'].append(None)
    #     self.structures_dict['voxel_idx'].append(None)

    # def modify_structure(self, structure=None, mask_3d=None):
    #     ind = self.structures_dict['name'].index(structure)
    #     self.structures_dict['structure_mask_3d'][ind] = mask_3d
    #     dose_vox_3d = self.structures_dict['structure_mask_3d'][ind] * \
    #                   self.opt_voxels_dict['ct_to_dose_voxel_map'][
    #                       0]
    #     self.structures_dict['voxel_idx'][ind] = np.unique(dose_vox_3d[dose_vox_3d > 0])

    def union(self, str_1: str, str_2: str, str1_or_str2: str):
        """
        :param str_1: structure name for the 1st structure
        :param str_2: structure name for the 2nd structure
        :param str1_or_str2: structure name for the union of  structure 1 and 2
        :return: create union of the structures and save it to structures list
        """
        if str1_or_str2 is None:
            raise Exception("str1_or_str2 need to be provided")
        ind1 = self.structures_dict['name'].index(str_1)
        ind2 = self.structures_dict['name'].index(str_2)
        mask_3d_1 = self.structures_dict['structure_mask_3d'][ind1]
        mask_3d_2 = self.structures_dict['structure_mask_3d'][ind2]
        new_mask_3d_1 = mask_3d_1 | mask_3d_2
        if str1_or_str2 not in self.structures_dict['name']:
            self.create_structure(new_structure=str1_or_str2, mask_3d=new_mask_3d_1)
        else:
            self.modify_structure(structure=str1_or_str2, mask_3d=new_mask_3d_1)

    def intersect(self, str_1: str, str_2: str, str1_and_str2: str):
        """
        :param str_1: structure name for the 1st structure
        :param str_2: structure name for the 2nd structure
        :param str1_and_str2: structure name for the intersection of  structure 1 and 2
        :return: create intersection of the structures and save it to structures list
        """
        if str1_and_str2 is None:
            raise Exception("str1_and_str2 need to be provided")
        ind1 = self.structures_dict['name'].index(str_1)
        ind2 = self.structures_dict['name'].index(str_2)
        mask_3d_1 = self.structures_dict['structure_mask_3d'][ind1]
        mask_3d_2 = self.structures_dict['structure_mask_3d'][ind2]
        new_mask_3d_1 = mask_3d_1 & mask_3d_2
        if str1_and_str2 not in self.structures_dict['name']:
            self.create_structure(new_structure=str1_and_str2, mask_3d=new_mask_3d_1)
        else:
            self.modify_structure(structure=str1_and_str2, mask_3d=new_mask_3d_1)

    def subtract(self, str_1: str, str_2: str, str1_sub_str2: str):
        """
        :param str_1: structure name for the 1st structure
        :param str_2: structure name for the 2nd structure
        :param str1_sub_str2: structure name for subtracting 2 from 1
        :return: create structure1 - structure2 and save it to list
        """
        if str1_sub_str2 is None:
            raise Exception("str1_sub_str2 need to be provided")
        ind1 = self.structures_dict['name'].index(str_1)
        ind2 = self.structures_dict['name'].index(str_2)
        mask_3d_1 = self.structures_dict['structure_mask_3d'][ind1]
        mask_3d_2 = self.structures_dict['structure_mask_3d'][ind2]
        new_mask_3d_1 = mask_3d_1 - mask_3d_2
        new_mask_3d_1[new_mask_3d_1 < 0] = np.uint8(0)
        if str1_sub_str2 not in self.structures_dict['name']:
            self.create_structure(new_structure=str1_sub_str2, mask_3d=new_mask_3d_1)
        else:
            self.modify_structure(structure=str1_sub_str2, mask_3d=new_mask_3d_1)
        # if new_structure_1 is not None:
        #     self.create_structure(new_structure=new_structure_1, mask_3d=new_mask_3d_1)
        # else:
        #     self.modify_structure(structure=structure_1, mask_3d=new_mask_3d_1)

    def down_sample_voxels(self, down_sample_xyz=None):
        if down_sample_xyz is None:
            down_sample_xyz = [3, 3, 1]
        vox_3d = self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        dose_to_ct_int = np.int(self.opt_voxels_dict['dose_voxel_resolution_xyz_mm'][0] / \
                                self.opt_voxels_dict['ct_voxel_resolution_xyz_mm'][0])
        all_vox = np.unique(vox_3d[vox_3d >= 0])
        vox_map = np.ones((len(all_vox), 2)) * np.int(-1)
        count = 0
        all_vox = np.argwhere(vox_3d >= 0)
        skip = []
        for ind_zyx in all_vox:
            # if not np.any(skip == ind_zyx):  # ind_zyx not in skip:
            if not np.any([np.all(ind_zyx == a) for a in skip]):
                patch = vox_3d[ind_zyx[0], ind_zyx[1]:ind_zyx[1] + (down_sample_xyz[-2] - 1) * dose_to_ct_int,
                        ind_zyx[2]:ind_zyx[2] + (down_sample_xyz[-3] - 1) * dose_to_ct_int]
                vox = np.unique(patch[patch >= 0])
                xy_ind = np.concatenate([np.argwhere(vox_3d[ind_zyx[0], :, :] == i) for i in vox])
                xyz_ind = np.column_stack((np.ones(len(xy_ind), dtype='int') * ind_zyx[0], xy_ind[:, :]))
                skip.extend(xyz_ind)
                vox_map[vox, 1] = count
                vox_map[vox, 0] = vox
                count = count + 1
        return vox_map

    # def structure(self, structure):
    #     st = Structure(structure, self)
    #     # self.members.append(person)
    #     return st
    @staticmethod
    def ball(n):
        struct = np.zeros((2 * n + 1, 2 * n + 1, 2 * n + 1), dtype=int)
        x, y, z = np.indices((2 * n + 1, 2 * n + 1, 2 * n + 1))
        mask = (x - n) ** 2 + (y - n) ** 2 + (z - n) ** 2 <= n ** 2
        struct[mask] = 1
        return struct

    @staticmethod
    def ellipsoid(n):  # n is number of pixels or voxels
        struct = np.zeros((2 * n[2] + 1, 2 * n[1] + 1, 2 * n[0] + 1), dtype=int)  # z y x
        x, y, z = np.indices((2 * n[2] + 1, 2 * n[1] + 1, 2 * n[0] + 1))
        mask = (x - n[2]) ** 2 / (n[2] ** 2) + (y - n[1]) ** 2 / (n[1] ** 2) + (z - n[0]) ** 2 / (n[0] ** 2) <= 1
        struct[mask] = 1
        return struct

    def expand(self, structure: str, margin_mm: float, new_structure: str):
        """
        :param structure: structure name to expand
        :param margin_mm: margin in mm
        :param new_structure: new structure name. if same as structure name,
        expand the same structure. else create new structure
        :return:
        """
        # from skimage import data, morphology, transform
        from scipy import ndimage
        ind = self.structures_dict['name'].index(structure)
        mask_3d = self.structures_dict['structure_mask_3d'][ind]

        # getting kernel size for expansion or shrinking
        num_voxels = np.round(margin_mm / np.asarray(self.ct_voxel_resolution_xyz_mm)).astype(int)
        # structure = morphology.square(2)
        # odd side length for our structuring elements, so that they are symmetric about a central pixel
        # struct = np.ones(
        #     (num_voxels[2]*2 + 1, num_voxels[1]*2 + 1, num_voxels[0]*2 + 1), dtype='int')  # 2n+1 for isotropic dilation or erosion
        # margin_mask_3d = ndimage.morphology.grey_dilation(mask_3d, structure=Structures.ellipsoid(num_voxels))
        # margin_mask_3d = ndimage.morphology.grey_dilation(mask_3d, size=np.shape(struct))
        # 3d kernel
        # struct = np.ones((3, 3, 3), dtype=int) # sqaure kernel
        # struct = ndimage.generate_binary_structure(rank=3, connectivity=1) # diamond kernel
        kernel = ndimage.generate_binary_structure(rank=3, connectivity=2)  # ball kernel
        # creating different iterations along z and xy directions
        num_iterations = np.int(num_voxels[0] / num_voxels[2])
        iterations_in_one_step = np.int(np.round(num_voxels[0] / num_iterations))
        if margin_mm > 0:
            # margin_mask_3d = ndimage.binary_dilation(mask_3d, structure=struct).astype(mask_3d.dtype)
            for i in range(num_iterations):
                if i == 0:
                    margin_mask_3d = ndimage.binary_dilation(mask_3d, structure=kernel,
                                                             iterations=iterations_in_one_step).astype(mask_3d.dtype)
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # pos = np.where(margin_mask_3d == 1)
                    # ax.scatter(pos[0], pos[1], pos[2], c='red')
                else:
                    flat = np.copy(kernel)
                    flat[0, :, :] = 0
                    flat[-1, :, :] = 0
                    margin_mask_3d = ndimage.binary_dilation(margin_mask_3d, structure=flat,
                                                             iterations=iterations_in_one_step).astype(mask_3d.dtype)
                    # pos = np.where(margin_mask_3d == 1)
                    # ax.scatter(pos[0], pos[1], pos[2], c='green')
            # testing of this idea
            # a = np.zeros((8, 10, 10), dtype=int)
            # a[3:4, 3:7, 3:7] = np.int(1)
            # b = ndimage.binary_dilation(a, structure=np.ones((3, 3, 3)), iterations=2).astype(a.dtype)
            # flat = np.ones((3, 3, 3), dtype=int)
            # flat[0, :, :] = 0
            # flat[-1, :, :] = 0
            # c = ndimage.binary_dilation(b, structure=flat.astype(int), iterations=1).astype(a.dtype)

            # margin_mask_3d = ndimage.binary_dilation(mask_3d, structure=structure).astype(mask_3d.dtype)
        if new_structure not in self.structures_dict['name']:
            self.create_structure(new_structure=new_structure, mask_3d=margin_mask_3d)
        else:
            self.modify_structure(structure=structure, mask_3d=margin_mask_3d)

    def shrink(self, structure, margin_mm=None, new_structure=None):
        """
            :param structure: structure name to shrink
            :param margin_mm: margin in mm
            :param new_structure: new structure name. if same as structure name,
            shrink the same structure. else create new structure
            :return:
        """
        # from skimage import data, morphology, transform
        from scipy import ndimage
        ind = self.structures_dict['name'].index(structure)
        mask_3d = self.structures_dict['structure_mask_3d'][ind]

        # getting kernel size for expansion or shrinking
        num_voxels = np.round(margin_mm / np.asarray(self.opt_voxels_dict['ct_voxel_resolution_xyz_mm'])).astype(int)
        struct = ndimage.generate_binary_structure(rank=3, connectivity=2)  # ball kernel
        # creating different iterations along z and xy directions
        num_iterations = np.int(num_voxels[0] / num_voxels[2])
        iterations_in_one_step = np.int(np.round(num_voxels[0] / num_iterations))
        for i in range(num_iterations):
            if i == 0:
                margin_mask_3d = ndimage.binary_erosion(mask_3d, structure=struct,
                                                        iterations=iterations_in_one_step).astype(mask_3d.dtype)
            else:
                flat = np.copy(struct)
                flat[0, :, :] = 0
                flat[-1, :, :] = 0
                margin_mask_3d = ndimage.binary_erosion(margin_mask_3d, structure=flat,
                                                        iterations=iterations_in_one_step).astype(mask_3d.dtype)

            # margin_mask_3d = ndimage.binary_dilation(mask_3d, structure=structure).astype(mask_3d.dtype)
        if new_structure not in self.structures_dict['name']:
            self.create_structure(new_structure=new_structure, mask_3d=margin_mask_3d)
        else:
            self.modify_structure(structure=structure, mask_3d=margin_mask_3d)
