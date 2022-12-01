import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import LinearRing, Polygon


class Structures:
    """
    A class representing structures.
    """

    def __init__(self, structures, opt_voxels):
        # super().__init__()
        self.structures_dict = structures
        self.opt_voxels_dict = opt_voxels
        self.preprocess_structures()

    # def __init__(self, *args):
    #     # super().__init__()
    #     if len(args) > 0 and isinstance(args[0], dict):
    #         self.structures_dict = args[0]
    #         self.opt_voxels_dict = args[1]
    #         self.preprocess_structures()
    #
    #     if isinstance(args[0], str):
    #         self.structure = args[0]

    def get_voxels_idx(self, structure_name):
        ind = self.structures_dict['name'].index(structure_name)
        vox_ind = self.structures_dict['voxel_idx'][ind]
        # vox_ind = np.where(self.opt_voxels_dict['voxel_structure_map'][0][:, ind] == 1)[0]
        return vox_ind

    def get_voxels_weights(self, structure_name):
        ind = self.structures_dict['name'].index(structure_name)
        vox_weights = self.structures_dict['voxel_weights'][ind]
        # vox_ind = np.where(self.opt_voxels_dict['voxel_structure_map'][0][:, ind] == 1)[0]
        return vox_weights

    def get_volume_cc(self, structure_name):
        ind = self.structures_dict['name'].index(structure_name)
        return self.structures_dict['volume_cc'][ind]

    def preprocess_structures(self, down_sample=False):
        self.structures_dict['voxel_idx'] = [None] * len(self.structures_dict['name'])
        self.structures_dict['voxel_weights'] = [None] * len(self.structures_dict['name'])
        for i in range(len(self.structures_dict['name'])):
            if down_sample:
                vox_3d = self.structures_dict['structure_mask_3d'][i] * self.opt_voxels_dict['ct_dose_map_down_sampled']
            else:
                vox_3d = self.structures_dict['structure_mask_3d'][i] * self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
            # self.structures_dict['voxel_idx'][i] = np.unique(vox_3d[vox_3d > 0])
            vox, counts = np.unique(vox_3d[vox_3d > 0], return_counts=True)
            self.structures_dict['voxel_idx'][i] = vox
            self.structures_dict['voxel_weights'][i] = counts / np.max(counts)  # calculate weight for each voxel

    # def add_subtract_margin(self, structure='PTV', margin_mm=5, new_structure=None):
    #     if new_structure is None:
    #         new_structure = 'NEW'
    #     ind = self.structures_dict['name'].index(structure)
    #     mask_3d = self.structures_dict['structure_mask_3d'][ind]
    #     margin_mask_3d = np.zeros_like(mask_3d, dtype='uint8')
    #     for i in range(mask_3d.shape[0]):
    #         slice_mask = mask_3d[i, :, :]
    #         # ret, binary = cv2.threshold(np.uint8(slice_mask), 0, 255, cv2.THRESH_BINARY_INV)
    #         contours, hierarchy = cv2.findContours(np.uint8(slice_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         if len(contours) > 0:
    #             slice_mask_margin = np.zeros_like(slice_mask, dtype='uint8')
    #             # cv2.drawContours(image=arr, contours=contours, contourIdx=-1, color=1, thickness=1)  # draw lat contour
    #             # cv2.imshow('orignial_image', arr)
    #             # arr = np.zeros_like(slice_mask, dtype='uint8')
    #             polygons = []
    #             for count_num in range(len(contours)):
    #                 pixel_coords = []
    #                 for j in contours[count_num]:
    #                     pixel_coords.append((j[0][0], j[0][1]))
    #                 r = LinearRing(pixel_coords)
    #                 s = Polygon(r)
    #                 shapely_poly = Polygon(s.buffer(margin_mm / self.opt_voxels_dict['ct_voxel_resolution_xyz_mm'][0]),
    #                                        [r])
    #                 polygon = np.array(list(shapely_poly.exterior.coords))
    #                 polygon = np.around([polygon]).astype(np.int32)
    #                 polygons.append(polygon)
    #             cv2.fillPoly(img=slice_mask_margin, pts=polygons, color=1)
    #             # slice_mask = slice_mask + arr
    #             # slice_mask[slice_mask > 0] = 1
    #             margin_mask_3d[i, :, :] = slice_mask_margin
    #
    #     # self.create_structure_data(new_structure=new_structure, mask_3d=margin_mask_3d)
    #     # self.structures_dict['structure_mask_3d'].append(margin_mask_3d)
    #     # self.structures_dict['name'].append(new_structure)
    #     # dose_vox_3d = self.structures_dict['structure_mask_3d'][-1] * self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
    #     # self.structures_dict['voxel_idx'].append(np.unique(dose_vox_3d[dose_vox_3d > 0]))
    #     return margin_mask_3d

    def create_structure(self, new_structure, mask_3d):
        for key in self.structures_dict.keys():
            if key == 'name':
                self.structures_dict['name'].append(new_structure)
            elif key == 'structure_mask_3d':
                self.structures_dict['structure_mask_3d'].append(mask_3d)
            elif key == 'voxel_idx':
                dose_vox_3d = self.structures_dict['structure_mask_3d'][-1] * \
                              self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
                # self.structures_dict['voxel_idx'].append(np.unique(dose_vox_3d[dose_vox_3d > 0]))
                vox, counts = np.unique(dose_vox_3d[dose_vox_3d > 0], return_counts=True)
                self.structures_dict['voxel_idx'].append(vox)
                self.structures_dict['voxel_weights'].append(counts / np.max(counts))
            elif key == 'voxel_weights':
                continue
            else:
                self.structures_dict[key].append(None)

    def modify_structure(self, structure, mask_3d):
        ind = self.structures_dict['name'].index(structure)
        for key in self.structures_dict.keys():
            if key == 'name':
                self.structures_dict['name'][ind] = structure
            elif key == 'structure_mask_3d':
                self.structures_dict['structure_mask_3d'][ind] = mask_3d
            elif key == 'voxel_idx':
                dose_vox_3d = self.structures_dict['structure_mask_3d'][ind] * \
                              self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
                # self.structures_dict['voxel_idx'][ind] = np.unique(dose_vox_3d[dose_vox_3d > 0])
                vox, counts = np.unique(dose_vox_3d[dose_vox_3d > 0], return_counts=True)
                self.structures_dict['voxel_idx'][ind] = vox
                self.structures_dict['voxel_weights'][ind] = counts / np.max(counts)
            elif key == 'voxel_weights':
                continue
            else:
                self.structures_dict[key][ind] = None

    def delete_structure(self, structure):
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

    def union(self, str_1=None, str_2=None, str1_or_str2=None):
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

    def intersect(self, str_1=None, str_2=None, str1_and_str2=None):
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

    def subtract(self, str_1, str_2, str1_sub_str2=None):
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

    def create_rinds(self, base_structure='PTV', size=None):

        if size is None:
            size = [5, 5, 20, 30, 500]
        print('creating rinds of size {} mm ..'.format(size))
        if isinstance(size, np.ndarray):
            size = list(size)
        for ind, s in enumerate(size):
            rind_name = 'RIND_{}'.format(ind)

            if ind == 0:
                self.expand(base_structure, margin_mm=s, new_structure=rind_name)
                self.subtract(rind_name, base_structure, str1_sub_str2=rind_name)
                # mask_3d_modified = self.add_subtract_margin(structure=base_structure, margin_mm=5, new_structure=rind_name)
                # self.subtract(structure_1=rind_name, structure_2=base_structure)
                # mask_3d_modified = self.structure(base_structure) + s
                # self.add_structure(rind_name, mask_3d_modified)
                # mask_3d_modified = self.structure(rind_name) - self.structure(base_structure)
                # self.modify_structure(rind_name, mask_3d_modified)
            else:
                prev_rind_name = 'RIND_{}'.format(ind - 1)
                self.expand(prev_rind_name, margin_mm=s, new_structure=rind_name)
                for j in range(ind):
                    rind_subtract = 'RIND_{}'.format(j)
                    if j == 0:
                        self.union(rind_subtract, base_structure, str1_or_str2='dummy')
                    else:
                        self.union(rind_subtract, 'dummy', str1_or_str2='dummy')
                self.subtract(rind_name, 'dummy', str1_sub_str2=rind_name)
                self.delete_structure('dummy')
                if ind == len(size)-1:
                    ct_to_dose_map = self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
                    dose_mask = ct_to_dose_map >= 0
                    dose_mask = dose_mask.astype(int)
                    self.create_structure('dose_mask', dose_mask)
                    self.intersect(rind_name, 'dose_mask', str1_and_str2=rind_name)
                    self.delete_structure('dose_mask')
        print('rinds created!!')
        # # self.add_subtract_margin(structure=prev_rind_name, margin_mm=s, new_structure=rind_name)
        # mask_3d_modified = self.structure(prev_rind_name) + s
        # self.add_structure(rind_name, mask_3d_modified)
        # mask_3d_modified = self.structure(rind_name) - self.structure(prev_rind_name)
        # self.modify_structure(rind_name, mask_3d_modified)
        # self.subtract(structure_1=rind_name, structure_2=prev_rind_name)

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
        struct = np.zeros((2 * n[2] + 1, 2 * n[1] + 1, 2 * n[0] + 1), dtype=int) # z y x
        x, y, z = np.indices((2 * n[2] + 1, 2 * n[1] + 1, 2 * n[0] + 1))
        mask = (x - n[2]) ** 2/(n[2]**2) + (y - n[1]) ** 2/(n[1]**2) + (z - n[0]) ** 2/(n[0]**2) <= 1
        struct[mask] = 1
        return struct

    def expand(self, structure, margin_mm=None, new_structure=None):
        from skimage import data, morphology, transform
        from scipy import ndimage
        ind = self.structures_dict['name'].index(structure)
        mask_3d = self.structures_dict['structure_mask_3d'][ind]

        # getting kernel size for expansion or shrinking
        num_voxels = np.round(margin_mm / np.asarray(self.opt_voxels_dict['ct_voxel_resolution_xyz_mm'])).astype(int)
        # structure = morphology.square(2)
        # odd side length for our structuring elements, so that they are symmetric about a central pixel
        # struct = np.ones(
        #     (num_voxels[2]*2 + 1, num_voxels[1]*2 + 1, num_voxels[0]*2 + 1), dtype='int')  # 2n+1 for isotropic dilation or erosion
        # margin_mask_3d = ndimage.morphology.grey_dilation(mask_3d, structure=Structures.ellipsoid(num_voxels))
        # margin_mask_3d = ndimage.morphology.grey_dilation(mask_3d, size=np.shape(struct))
        # 3d kernel
        # struct = np.ones((3, 3, 3), dtype=int) # sqaure kernel
        # struct = ndimage.generate_binary_structure(rank=3, connectivity=1) # diamond kernel
        struct = ndimage.generate_binary_structure(rank=3, connectivity=2)  # ball kernel
        # creating different iterations along z and xy directions
        num_iterations = np.int(num_voxels[0] / num_voxels[2])
        iterations_in_one_step = np.int(np.round(num_voxels[0] / num_iterations))
        if margin_mm > 0:
            # margin_mask_3d = ndimage.binary_dilation(mask_3d, structure=struct).astype(mask_3d.dtype)
            for i in range(num_iterations):
                if i == 0:
                    margin_mask_3d = ndimage.binary_dilation(mask_3d, structure=struct,
                                                             iterations=iterations_in_one_step).astype(mask_3d.dtype)
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # pos = np.where(margin_mask_3d == 1)
                    # ax.scatter(pos[0], pos[1], pos[2], c='red')
                else:
                    flat = np.copy(struct)
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
        from skimage import data, morphology, transform
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



# class Structure:
#
#     def __init__(self, structure, structures):
#         self.structure = structure
#         self.structures = structures
#
#     def __add__(self, other):
#         if type(other) is int:
#             margin_mask_3d = self.expand_shrink(margin=other)
#             self.mask_3d = margin_mask_3d
#         else:
#             ind1 = self.structures.structures_dict['name'].index(self.structure)
#             ind2 = other.structures.structures_dict['name'].index(other.structure)
#             mask_3d_1 = self.structures.structures_dict['structure_mask_3d'][ind1]
#             mask_3d_2 = other.structures.structures_dict['structure_mask_3d'][ind2]
#             margin_mask_3d = mask_3d_1 | mask_3d_2
#             new_object = Structure(self.structures.structures_dict['name'])
#             # self.structures.structures_dict['structure_mask_3d'][ind1] = new_mask_3d_1
#         return margin_mask_3d
#
#     def __sub__(self, other):
#         if type(other) is int:
#             new_mask_3d_1 = self.expand_shrink(margin=-other)
#         else:
#             ind1 = self.structures.structures_dict['name'].index(self.structure)
#             ind2 = other.structures.structures_dict['name'].index(other.structure)
#             mask_3d_1 = self.structures.structures_dict['structure_mask_3d'][ind1]
#             mask_3d_2 = other.structures.structures_dict['structure_mask_3d'][ind2]
#             new_mask_3d_1 = mask_3d_1 - mask_3d_2
#             new_mask_3d_1[new_mask_3d_1 < 0] = np.uint8(0)
#         # self.structures.structures_dict['structure_mask_3d'][ind1] = new_mask_3d_1
#         return new_mask_3d_1
#
#     def __mul__(self, other):
#         ind1 = self.structures.structures_dict['name'].index(self.structure)
#         ind2 = other.structures.structures_dict['name'].index(other.structure)
#         mask_3d_1 = self.structures.structures_dict['structure_mask_3d'][ind1]
#         mask_3d_2 = other.structures.structures_dict['structure_mask_3d'][ind2]
#         new_mask_3d_1 = mask_3d_1 & mask_3d_2
#         # new_mask_3d_1[new_mask_3d_1 < 0] = np.uint8(0)
#         # self.structures.structures_dict['structure_mask_3d'][ind1] = new_mask_3d_1
#         return new_mask_3d_1
#
#     def expand_shrink(self, margin):
#         ind = self.structures.structures_dict['name'].index(self.structure)
#         mask_3d = self.structures.structures_dict['structure_mask_3d'][ind]
#         margin_mask_3d = np.zeros_like(mask_3d, dtype='uint8')
#         for i in range(mask_3d.shape[0]):
#             slice_mask = mask_3d[i, :, :]
#             # ret, binary = cv2.threshold(np.uint8(slice_mask), 0, 255, cv2.THRESH_BINARY_INV)
#             contours, hierarchy = cv2.findContours(np.uint8(slice_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             if len(contours) > 0:
#                 slice_mask_margin = np.zeros_like(slice_mask, dtype='uint8')
#                 # cv2.drawContours(image=arr, contours=contours, contourIdx=-1, color=1, thickness=1)  # draw lat contour
#                 # cv2.imshow('orignial_image', arr)
#                 # arr = np.zeros_like(slice_mask, dtype='uint8')
#                 polygons = []
#                 for count_num in range(len(contours)):
#                     pixel_coords = []
#                     for j in contours[count_num]:
#                         pixel_coords.append((j[0][0], j[0][1]))
#                     r = LinearRing(pixel_coords)
#                     s = Polygon(r)
#                     shapely_poly = Polygon(
#                         s.buffer(
#                             margin / self.structures.opt_voxels_dict['ct_voxel_resolution_xyz_mm'][0]),
#                         [r])
#                     polygon = np.array(list(shapely_poly.exterior.coords))
#                     polygon = np.around([polygon]).astype(np.int32)
#                     polygons.append(polygon)
#                 cv2.fillPoly(img=slice_mask_margin, pts=polygons, color=1)
#                 # slice_mask = slice_mask + arr
#                 # slice_mask[slice_mask > 0] = 1
#                 margin_mask_3d[i, :, :] = slice_mask_margin
#         return margin_mask_3d
