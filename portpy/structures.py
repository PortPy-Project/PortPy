import numpy as np
from copy import deepcopy


class Structures:
    """
    A class representing structures.

    - **Attributes** ::

        :param structures_dict: structure dictionary that contains information about the structures present in the patient's CT scan.
        :type structures_dict: dict
        :param opt_voxels_dict: It contains information about optimization voxels in the form of dictionary
        :type opt_voxels_dict: dict


    - **Methods** ::

        :get_volume_cc(struct):
            Get volume in cc for the structure
        :expand(structure, margin_mm, new_structure):
            Expand the structure with margin_mm in mm
        :shrink(structure, margin_mm, new_structure):
            Shrink the structure with margin_mm in mm
        :union(str_1, str_2, str1_or_str2):
            Create union of two structures str_1 and str_2
        :intersect(str_1, str_2, str1_and_str2):
            Create intersect of two structures str_1 and str_2


    """

    def __init__(self, structures: dict, opt_voxels: dict) -> None:
        """

        :param structures: structures dictionary that contains information about the structures present in the patient's CT scan.
        :param opt_voxels: optimization voxels dictionary containing data about optimization voxels
        """
        self.structures_dict = structures
        self.opt_voxels_dict = opt_voxels
        self.opt_voxels_dict['name'] = structures['name']
        self._ct_voxel_resolution_xyz_mm = deepcopy(self.opt_voxels_dict['ct_voxel_resolution_xyz_mm'])
        self.preprocess_structures()

    def get_volume_cc(self, structure_name: str):
        """
        Get volume in cc for the structure

         :param structure_name: name of the structure in plan
         :return: volume of the structure
         """
        ind = self.structures_dict['name'].index(structure_name)
        return self.structures_dict['volume_cc'][ind]

    def preprocess_structures(self):
        """
        preprocess structures to create optimization voxel indices for the structure
        :return:
        """
        self.opt_voxels_dict['voxel_idx'] = [None] * len(self.structures_dict['name'])
        self.opt_voxels_dict['voxel_size'] = [None] * len(self.structures_dict['name'])
        for i in range(len(self.structures_dict['name'])):

            vox_3d = self.structures_dict['structure_mask_3d'][i] * self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
            # my_plan.structures_dict['voxel_idx'][i] = np.unique(vox_3d[vox_3d > 0])
            vox, counts = np.unique(vox_3d[vox_3d > 0], return_counts=True)
            self.opt_voxels_dict['voxel_idx'][i] = vox
            self.opt_voxels_dict['voxel_size'][i] = counts / np.max(counts)  # calculate weight for each voxel

    def create_structure(self, new_structure: str, mask_3d: np.ndarray) -> None:
        """
        Create a new structure and append its mask to the structures_dict

        :param new_structure: name of the new structure
        :param mask_3d: 3d mask for the structure
        :return: create new_structure
        """
        for key in self.structures_dict.keys():
            if key == 'name':
                self.structures_dict['name'].append(new_structure)
            elif key == 'structure_mask_3d':
                self.structures_dict['structure_mask_3d'].append(mask_3d)
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

    def union(self, str_1: str, str_2: str, str1_or_str2: str) -> None:
        """
        Create union of two structures str_1 and str_2. If str1_or_str2 is not in structures dict,
        it will create new structures. If str1_or_str2 is in structure_dict, it will modify the structure

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

    def intersect(self, str_1: str, str_2: str, str1_and_str2: str) -> None:
        """
        Create intersection of two structures str_1 and str_2. If str1_and_str2 is not in structures dict,
        it will create new structures. If str1_and_str2 is in structure_dict, it will modify the structure

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

    def subtract(self, str_1: str, str_2: str, str1_sub_str2: str) -> None:
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

    def expand(self, structure: str, margin_mm: float, new_structure: str) -> None:
        """

        Expand the structure with the given margin_mm.

        :param structure: structure name to expand
        :param margin_mm: margin_mm in mm
        :param new_structure: new structure name. if same as structure name,
        expand the same structure. else create new structure
        :return: expand and save the structure mask in structures dictionary
        """
        # from skimage import data, morphology, transform
        from scipy import ndimage
        ind = self.structures_dict['name'].index(structure)
        mask_3d = self.structures_dict['structure_mask_3d'][ind]

        # getting kernel size for expansion or shrinking
        num_voxels = np.round(margin_mm / np.asarray(self._ct_voxel_resolution_xyz_mm)).astype(int)

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
                else:
                    flat = np.copy(kernel)
                    flat[0, :, :] = 0
                    flat[-1, :, :] = 0
                    margin_mask_3d = ndimage.binary_dilation(margin_mask_3d, structure=flat,
                                                             iterations=iterations_in_one_step).astype(mask_3d.dtype)
        if new_structure not in self.structures_dict['name']:
            self.create_structure(new_structure=new_structure, mask_3d=margin_mask_3d)
        else:
            self.modify_structure(structure=structure, mask_3d=margin_mask_3d)

    def shrink(self, structure: str, margin_mm: float, new_structure: str) -> None:
        """
        Shrink the structure with margin_mm.

            :param structure: structure name to shrink
            :param margin_mm: margin_mm in mm
            :param new_structure: new structure name. if same as structure, shrink the same structure. else create new structure
            :return: shrink and save the structure in structures_dictionary

        """

        from scipy import ndimage
        ind = self.structures_dict['name'].index(structure)
        mask_3d = self.structures_dict['structure_mask_3d'][ind]

        # getting kernel size for expansion or shrinking
        num_voxels = np.round(margin_mm / np.asarray(self._ct_voxel_resolution_xyz_mm)).astype(int)
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
