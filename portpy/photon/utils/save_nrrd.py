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

from __future__ import annotations
import SimpleITK as sitk
from pathlib import Path
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from portpy.photon.plan import Plan
from portpy.photon.ct import CT
from portpy.photon.structures import Structures


def save_nrrd(my_plan: Plan = None, sol: dict = None, dose_1d: np.ndarray = None, data_dir: str = None, ct: CT = None,
              structs: Structures = None, ct_filename: str = 'ct', dose_filename: str = 'dose',
              rt_struct_filename: str = 'rtss') -> None:
    """
    save nrrd in the path directory else save in patient data directory

    :param my_plan: object of class Plan
    :param sol: optimal solution dict
    :param dose_1d: dose as 1d array
    :param data_dir: save nrrd images of ct, dose_1d and struct_name set in path directory
    :param ct: object of class CT
    :param structs: object of class structs
    :param ct_filename: ct file name
    :param dose_filename: dose file name
    :param rt_struct_filename: rt_struct file name
    :return: save nrrd images in path
    """
    import os
    if ct is None:
        ct = my_plan.ct
    if structs is None:
        structs = my_plan.structures
    if data_dir is None:
        data_dir = os.path.join(Path(__file__).parents[2], 'data', ct.patient_id)
    elif not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ct_arr = ct.ct_dict['ct_hu_3d'][0]
    ct_image = sitk.GetImageFromArray(ct_arr)
    ct_image.SetOrigin(ct.ct_dict['origin_xyz_mm'])
    ct_image.SetSpacing(ct.ct_dict['resolution_xyz_mm'])
    ct_image.SetDirection(ct.ct_dict['direction'])
    sitk.WriteImage(ct_image, os.path.join(data_dir, ct_filename + '.nrrd'))

    dose_arr = []
    if sol is not None:
        dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity']*my_plan.get_num_of_fractions())
        dose_arr = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
    else:
        dose_arr = my_plan.inf_matrix.dose_1d_to_3d(dose_1d=dose_1d)
    dose = sitk.GetImageFromArray(dose_arr)
    dose.SetOrigin(ct.ct_dict['origin_xyz_mm'])
    dose.SetSpacing(ct.ct_dict['resolution_xyz_mm'])
    dose.SetDirection(ct.ct_dict['direction'])
    sitk.WriteImage(dose, os.path.join(data_dir, dose_filename + '.nrrd'))

    labels = structs.structures_dict['structure_mask_3d']
    mask_arr = np.array(labels).transpose((1, 2, 3, 0))
    mask = sitk.GetImageFromArray(mask_arr.astype('uint8'))
    # for i, struct_name in enumerate(my_plan.structures.structures_dict['name']):
    #     segment_name = "Segment{0}_Name".format(i)
    #     mask.SetMetaData(segment_name, struct_name)
    mask.SetOrigin(ct.ct_dict['origin_xyz_mm'])
    mask.SetSpacing(ct.ct_dict['resolution_xyz_mm'])
    mask.SetDirection(ct.ct_dict['direction'])
    sitk.WriteImage(mask, os.path.join(data_dir, rt_struct_filename + '.seg.nrrd'), True)
    # my_plan.visualize.patient_name = my_plan.patient_name
