# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use — including offering PortPy as a service,
# or incorporating it into a commercial product — requires
# a separate commercial license.
# ----------------------------------------------------------------------

import numpy as np
from copy import deepcopy
from .data_explorer import DataExplorer


class Structures:
    """
    A class representing structures.

    - **Attributes** ::

        :param structures_dict: struct_name dictionary that contains information about the structures present in the patient's CT scan.
        :type structures_dict: dict
        :param opt_voxels_dict: It contains information about optimization voxels in the form of dictionary
        :type opt_voxels_dict: dict


    - **Methods** ::

        :get_volume_cc(struct):
            Get volume in cc for the struct_name
        :expand(struct_name, margin_mm, new_struct_name):
            Expand the struct_name with margin_mm in mm
        :shrink(struct_name, margin_mm, new_struct_name):
            Shrink the struct_name with margin_mm in mm
        :union(struct_1_name, struct_2_name, new_struct_name):
            Create union of two structures struct_1_name and struct_2_name
        :intersect(struct_1_name, struct_2_name, new_struct_name):
            Create intersect of two structures struct_1_name and struct_2_name


    """

    def __init__(self, data: DataExplorer) -> None:
        """

        :param data: object of DataExplorer Class
        """
        metadata = data.load_metadata()
        structures_dict = data.load_data(meta_data=metadata['structures'])
        opt_voxels_dict = data.load_data(meta_data=metadata['opt_voxels'])
        self.structures_dict = structures_dict
        self.opt_voxels_dict = opt_voxels_dict
        self.opt_voxels_dict['name'] = structures_dict['name']
        self._ct_voxel_resolution_xyz_mm = deepcopy(self.opt_voxels_dict['ct_voxel_resolution_xyz_mm'])
        self.preprocess_structures()
        self.patient_id = data.patient_id

    def get_structures(self) -> list:
        """
        Returns all the struct_name names as list
        :return:
        """
        return self.structures_dict['name']

    def get_volume_cc(self, structure_name: str):
        """
        Get volume in cc for the struct_name

         :param structure_name: name of the struct_name in plan
         :return: volume of the struct_name
         """
        ind = self.structures_dict['name'].index(structure_name)
        return self.structures_dict['volume_cc'][ind]

    def get_fraction_of_vol_in_calc_box(self, structure_name: str):
        """
        Get fraction of volume in calc box for the struct_name

         :param structure_name: name of the struct_name in plan
         :return: volume of the struct_name
         """
        ind = self.structures_dict['name'].index(structure_name)
        return self.structures_dict['fraction_of_vol_in_calc_box'][ind]

    def get_structure_mask_3d(self, structure_name: str):
        """
        Get fraction of volume in calc box for the struct_name

         :param structure_name: name of the struct_name in plan
         :return: volume of the struct_name
         """
        ind = self.structures_dict['name'].index(structure_name)
        return self.structures_dict['structure_mask_3d'][ind]

    def union(self, struct_1_name: str, struct_2_name: str, new_struct_name: str = None, return_mask=False):
        """
        Create union of two structures struct_1_name and struct_2_name. If str1_or_str2 is not in structures dict,
        it will create new structures. If str1_or_str2 is in structure_dict, it will modify the struct_name

        :param struct_1_name: struct_name name for the 1st struct_name
        :param struct_2_name: struct_name name for the 2nd struct_name
        :param new_struct_name: struct_name name for the union of  struct_name 1 and 2
        :return: create union of the structures and save it to structures list
        """

        ind1 = self.structures_dict['name'].index(struct_1_name)
        ind2 = self.structures_dict['name'].index(struct_2_name)
        mask_3d_1 = self.structures_dict['structure_mask_3d'][ind1]
        mask_3d_2 = self.structures_dict['structure_mask_3d'][ind2]
        new_mask_3d_1 = mask_3d_1 | mask_3d_2

        if new_struct_name is not None:
            if new_struct_name not in self.structures_dict['name']:
                self.create_structure(new_struct_name=new_struct_name, mask_3d=new_mask_3d_1)
            else:
                self.modify_structure(struct_name=new_struct_name, mask_3d=new_mask_3d_1)
        if return_mask:
            return new_mask_3d_1

    def intersect(self, struct_1_name: str, struct_2_name: str, new_struct_name: str = None, return_mask=False):
        """
        Create intersection of two structures struct_1_name and struct_2_name. If str1_and_str2 is not in structures dict,
        it will create new structures. If str1_and_str2 is in structure_dict, it will modify the struct_name

        :param struct_1_name: struct_name name for the 1st struct_name
        :param struct_2_name: struct_name name for the 2nd struct_name
        :param new_struct_name: struct_name name for the intersection of  struct_name 1 and 2
        :param return_mask: return 3d mask
        :return: create intersection of the structures and save it to structures list
        """

        ind1 = self.structures_dict['name'].index(struct_1_name)
        ind2 = self.structures_dict['name'].index(struct_2_name)
        mask_3d_1 = self.structures_dict['structure_mask_3d'][ind1]
        mask_3d_2 = self.structures_dict['structure_mask_3d'][ind2]
        new_mask_3d_1 = mask_3d_1 & mask_3d_2

        if new_struct_name not in self.structures_dict['name']:
            self.create_structure(new_struct_name=new_struct_name, mask_3d=new_mask_3d_1)
        else:
            self.modify_structure(struct_name=new_struct_name, mask_3d=new_mask_3d_1)
        if return_mask:
            return new_mask_3d_1

    def subtract(self, struct_1_name: str, struct_2_name: str, new_struct_name: str = None, return_mask=False):
        """
        :param struct_1_name: struct_name name for the 1st struct_name
        :param struct_2_name: struct_name name for the 2nd struct_name
        :param new_struct_name: struct_name name for subtracting 2 from 1
        :param return_mask: return 3d mask of the structure
        :return: create structure1 - structure2 and save it to list
        """

        ind1 = self.structures_dict['name'].index(struct_1_name)
        ind2 = self.structures_dict['name'].index(struct_2_name)
        mask_3d_1 = deepcopy(self.structures_dict['structure_mask_3d'][ind1])
        mask_3d_2 = deepcopy(self.structures_dict['structure_mask_3d'][ind2])
        new_mask_3d_1 = mask_3d_1.astype(int) - mask_3d_2.astype(int)  # convert to integer before doing the operation
        new_mask_3d_1[new_mask_3d_1 < 0] = int(0)
        new_mask_3d_1.astype('uint8')
        if new_struct_name is not None:
            if new_struct_name not in self.structures_dict['name']:
                self.create_structure(new_struct_name=new_struct_name, mask_3d=new_mask_3d_1)
            else:
                self.modify_structure(struct_name=new_struct_name, mask_3d=new_mask_3d_1)
        if return_mask:
            return new_mask_3d_1

    def expand(self, struct_name: str = None, margin_mm: float = None, new_struct_name: str = None,
               return_mask: bool = False, mask_3d: np.ndarray = None):
        """

        Expand the struct_name with the given margin_mm.

        :param struct_name: struct_name name to expand
        :param margin_mm: margin_mm in mm
        :param new_struct_name: new struct_name name. if same as struct_name name,
        expand the same struct_name. else create new struct_name
        :return: expand and save the struct_name mask in structures dictionary
        """
        # from skimage import data, morphology, transform
        from scipy import ndimage
        if mask_3d is None:
            ind = self.structures_dict['name'].index(struct_name)
            mask_3d = self.structures_dict['structure_mask_3d'][ind]

        # getting kernel size for expansion or shrinking
        if margin_mm > 0:
            num_voxels = np.round(margin_mm / np.asarray(self._ct_voxel_resolution_xyz_mm)).astype(int)

            kernel = ndimage.generate_binary_structure(rank=3, connectivity=2)  # ball kernel
            # creating different iterations along z and xy directions
            num_iterations = int(num_voxels[0] / num_voxels[2])
            iterations_in_one_step = int(np.round(num_voxels[0] / num_iterations))
            # margin_mask_3d = ndimage.binary_dilation(mask_3d, struct_name=struct).astype(mask_3d.dtype)
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
        elif margin_mm == 0:
            margin_mask_3d = mask_3d
        else:
            raise ValueError('Invalid margin {}'.format(margin_mm))
        if new_struct_name is not None:
            if new_struct_name not in self.structures_dict['name']:
                self.create_structure(new_struct_name=new_struct_name, mask_3d=margin_mask_3d)
            else:
                self.modify_structure(struct_name=struct_name, mask_3d=margin_mask_3d)
        if return_mask:
            return margin_mask_3d

    def shrink(self, struct_name: str = None, margin_mm: float = None, new_struct_name: str = None,
               return_mask: bool = False, mask_3d: np.ndarray = None):
        """
        Shrink the struct_name with margin_mm.

            :param struct_name: struct_name name to shrink
            :param margin_mm: margin_mm in mm
            :param new_struct_name: new struct_name name. if same as struct_name, shrink the same struct_name. else create new struct_name
            :return: shrink and save the struct_name in structures_dictionary

        """

        from scipy import ndimage
        if mask_3d is None:
            ind = self.structures_dict['name'].index(struct_name)
            mask_3d = self.structures_dict['structure_mask_3d'][ind]


        # getting kernel size for expansion or shrinking
        num_voxels = np.round(margin_mm / np.asarray(self._ct_voxel_resolution_xyz_mm)).astype(int)
        struct = ndimage.generate_binary_structure(rank=3, connectivity=2)  # ball kernel
        # creating different iterations along z and xy directions
        num_iterations = int(num_voxels[0] / num_voxels[2])
        iterations_in_one_step = int(np.round(num_voxels[0] / num_iterations))
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

            # margin_mask_3d = ndimage.binary_dilation(mask_3d, struct_name=struct_name).astype(mask_3d.dtype)
        if new_struct_name is not None:
            if new_struct_name not in self.structures_dict['name']:
                self.create_structure(new_struct_name=new_struct_name, mask_3d=margin_mask_3d)
            else:
                self.modify_structure(struct_name=struct_name, mask_3d=margin_mask_3d)
        if return_mask:
            return margin_mask_3d

    def create_opt_structures(self, opt_params=None, clinical_criteria=None):
        # create rinds for optimization

        obj_funcs = []
        opt_params_constraints = []
        criteria = []
        if opt_params is not None:
            obj_funcs = opt_params['objective_functions'] if 'objective_functions' in opt_params else []
            opt_params_constraints = opt_params['constraints'] if 'constraints' in opt_params else []

        if clinical_criteria is not None:
            criteria = clinical_criteria.clinical_criteria_dict['criteria']

        constraints = criteria + opt_params_constraints
        print('Creating optimization structures.. It may take some time due to dilation')
        for ind, obj in enumerate(obj_funcs):
            if 'structure_def' in obj:
                if obj['structure_name'] not in self.get_structures():
                    structure_def = obj['structure_def']
                    try:
                        mask_3d = self.evaluate_expression(structure_def)
                        result = mask_3d & self.get_structure_mask_3d('BODY')
                        self.create_structure(new_struct_name=obj['structure_name'], mask_3d=result)
                    except:
                        print('Cannot evaluate structure defintion {}'.format(structure_def))

        for ind, criterion in enumerate(constraints):
            if 'structure_def' in criterion['parameters']:
                param = criterion['parameters']
                structure_def = param['structure_def']
                if param['structure_name'] not in self.get_structures():
                    try:
                        mask_3d = self.evaluate_expression(structure_def)
                        result = mask_3d & self.get_structure_mask_3d('BODY')
                        self.create_structure(new_struct_name=param['structure_name'], mask_3d=result)
                    except:
                        print('Cannot evaluate structure defintion {}'.format(structure_def))
        print('Optimization structures created!!')
        # for param in rind_params:
        #     self.set_opt_voxel_idx(struct_name=param['name'])
        self.preprocess_structures()

    def preprocess_structures(self):
        """
        preprocess structures to create optimization voxel indices for the struct_name
        :return:
        """
        self.opt_voxels_dict['voxel_idx'] = [None] * len(self.structures_dict['name'])
        self.opt_voxels_dict['voxel_volume_cc'] = [None] * len(self.structures_dict['name'])
        for i in range(len(self.structures_dict['name'])):
            vox_3d = self.structures_dict['structure_mask_3d'][i] * self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
            # my_plan.structures_dict['voxel_idx'][i] = np.unique(vox_3d[vox_3d > 0])
            vox, counts = np.unique(vox_3d[vox_3d > 0], return_counts=True)
            self.opt_voxels_dict['voxel_idx'][i] = vox
            self.opt_voxels_dict['voxel_volume_cc'][i] = counts * np.prod(
                self._ct_voxel_resolution_xyz_mm)/1000  # calculate weight for each voxel
            # dividing by 1000 due to conversion from mm3 to cm3
            # self.opt_voxels_dict['voxel_volume_cc'][i] = counts / np.max(counts)  # calculate weight for each voxel

    def create_structure(self, new_struct_name: str, mask_3d: np.ndarray) -> None:
        """
        Create a new struct_name and append its mask to the structures_dict

        :param new_struct_name: name of the new struct_name
        :param mask_3d: 3d mask for the struct_name
        :return: create new_struct_name
        """
        for key in self.structures_dict.keys():
            if key == 'name':
                self.structures_dict['name'].append(new_struct_name)
            elif key == 'structure_mask_3d':
                self.structures_dict['structure_mask_3d'].append(mask_3d)
            elif key == 'volume_cc':
                counts = np.count_nonzero(mask_3d)
                self.structures_dict['volume_cc'].append(counts * np.prod(self._ct_voxel_resolution_xyz_mm) / 1000)
            elif key == 'fraction_of_vol_in_calc_box':  # implement it..
                counts = np.count_nonzero(mask_3d)
                volume_cc = counts * np.prod(self._ct_voxel_resolution_xyz_mm) / 1000
                ct_to_dose_map = self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
                dose_mask = ct_to_dose_map >= 0
                dose_mask = dose_mask.astype('uint8')
                frac_of_mask_in_calc_box = mask_3d & dose_mask
                counts = np.count_nonzero(frac_of_mask_in_calc_box)
                volume_cc_in_calc_box = counts * np.prod(self._ct_voxel_resolution_xyz_mm) / 1000
                if volume_cc == 0:
                    self.structures_dict['fraction_of_vol_in_calc_box'].append(0)
                else:
                    self.structures_dict['fraction_of_vol_in_calc_box'].append(volume_cc_in_calc_box/volume_cc) # avoid divide by zero error
            else:
                self.structures_dict[key].append(None)

    def modify_structure(self, struct_name: str, mask_3d: np.ndarray) -> None:
        """
        :param struct_name: name of the struct_name to be modified
        :param mask_3d: 3d mask for the struct_name
        :return: modify struct_name
        """
        ind = self.structures_dict['name'].index(struct_name)
        for key in self.structures_dict.keys():
            if key == 'name':
                self.structures_dict['name'][ind] = struct_name
            elif key == 'structure_mask_3d':
                self.structures_dict['structure_mask_3d'][ind] = mask_3d
            elif key == 'volume_cc':
                counts = np.count_nonzero(mask_3d)
                self.structures_dict['volume_cc'][ind] = (counts * np.prod(self._ct_voxel_resolution_xyz_mm) / 1000)
            elif key == 'fraction_of_vol_in_calc_box':  # implement it..
                counts = np.count_nonzero(mask_3d)
                volume_cc = counts * np.prod(self._ct_voxel_resolution_xyz_mm) / 1000
                ct_to_dose_map = self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
                dose_mask = ct_to_dose_map >= 0
                dose_mask = dose_mask.astype('uint8')
                frac_of_mask_in_calc_box = mask_3d & dose_mask
                counts = np.count_nonzero(frac_of_mask_in_calc_box)
                volume_cc_in_calc_box = counts * np.prod(self._ct_voxel_resolution_xyz_mm) / 1000
                self.structures_dict['fraction_of_vol_in_calc_box'][ind] = volume_cc_in_calc_box/volume_cc
            else:
                self.structures_dict[key][ind] = None

    def delete_structure(self, structure: str):
        """
        :param structure: struct_name to be removed
        :return:
        """
        ind = self.structures_dict['name'].index(structure)
        for key in self.structures_dict.keys():
            del self.structures_dict[key][ind]

    def get_ct_voxel_resolution_xyz_mm(self):
        return self._ct_voxel_resolution_xyz_mm

    def set_opt_voxel_idx(self, struct_name):
        ind = self.structures_dict['name'].index(struct_name)
        vox_3d = self.structures_dict['structure_mask_3d'][ind] * \
                 self.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        # my_plan.structures_dict['voxel_idx'][i] = np.unique(vox_3d[vox_3d > 0])
        vox, counts = np.unique(vox_3d[vox_3d > 0], return_counts=True)
        self.opt_voxels_dict['voxel_idx'].append(vox)
        # self.opt_voxels_dict['voxel_volume_cc'].append(counts / np.max(counts))  # calculate weight for each voxel
        self.opt_voxels_dict['voxel_volume_cc'].append(
            counts * np.prod(self._ct_voxel_resolution_xyz_mm))  # calculate weight for each voxel
        self.opt_voxels_dict['name'].append(struct_name)

    def apply_operator(self, operators_stack, values_stack):
        operator = operators_stack.pop()
        right_operand = values_stack.pop()
        if isinstance(right_operand, str):
            if right_operand == "inf":
                right_operand = "500"
        left_operand = values_stack.pop()

        if isinstance(left_operand, str):
            if not left_operand.isnumeric():
                if left_operand not in self.structures_dict['name']:
                    raise Exception("Invalid structure {}".format(left_operand))
                else:
                    left_operand = self.get_structure_mask_3d(left_operand)
        if isinstance(right_operand, str):
            if not right_operand.isnumeric():
                if right_operand not in self.structures_dict['name']:
                    raise Exception("Invalid structure {}".format(right_operand))
                else:
                    right_operand = self.get_structure_mask_3d(right_operand)

        if operator == '+':
            if isinstance(right_operand, str):
                result = self.expand(mask_3d=left_operand, margin_mm=float(right_operand), return_mask=True)
            else:
                result = left_operand | right_operand
        elif operator == '-':
            if isinstance(right_operand, str):
                result = self.shrink(mask_3d=left_operand, margin_mm=float(right_operand), return_mask=True)
            else:
                result = left_operand.astype(int) - right_operand.astype(int)
                result[result < 0] = int(0)
                result.astype('uint8')

        elif operator == '|':
            result = left_operand | right_operand
        elif operator == '&':
            result = left_operand & right_operand

        values_stack.append(result)

    def evaluate_expression(self, expression):

        operators_stack = []
        values_stack = []

        i = 0
        while i < len(expression):
            if expression[i] == ' ':
                i += 1
                continue
            elif expression[i].isalpha() or expression[i].isdigit():
                j = i
                while j < len(expression) and (expression[j].isalpha() or expression[j] in '-_.$^' or expression[j].isdigit()):
                    if not ((j > 0 and expression[j - 1].isspace()) and (j + 1 < len(expression) and expression[j + 1].isspace())):
                        j += 1
                struct = expression[i:j]
                values_stack.append(struct)
                i = j - 1
            elif expression[i] in '+-|&':
                while operators_stack and operators_stack[-1] in '+-|&':
                    try:
                        self.apply_operator(operators_stack, values_stack)
                    except:
                        raise Exception("Cannot evaluate expression")
                operators_stack.append(expression[i])
            elif expression[i] == '(':
                operators_stack.append(expression[i])
            elif expression[i] == ')':
                while operators_stack[-1] != '(':
                    self.apply_operator(operators_stack, values_stack)
                operators_stack.pop()

            i += 1

        while operators_stack:
            try:
                self.apply_operator(operators_stack, values_stack)
            except:
                raise Exception("Cannot evaluate expression")

        return values_stack[0]
