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


def view_in_slicer_jupyter(my_plan, dose_1d: np.ndarray = None, sol: dict = None,
                           ct_name: str = 'ct', dose_name: str = 'dose', struct_set_name: str = 'rt_struct',
                           show_ct: bool = True,
                           show_dose: bool = True,
                           show_structs: bool = True):
    """
    This method helps to visualize CT, Dose and Rt struct in 3d slicer


    :param my_plan: object of class plan
    :param dose_1d: dose in 1d
    :param sol: solution dictionary
    :param ct_name: Default to 'ct'. name of the ct node in 3D slicer
    :param dose_name: Default to 'dose'. name of the dose node in 3D slicer
    :param struct_set_name: name of the rtstruct
    :param show_structs: default to True. If false, will not create struct_name node
    :param show_dose: default to True. If false, will not create dose node
    :param show_ct:default to True. If false, will not create ct node
    :return: visualize in slicer jupyter
    """
    try:
        import slicer
    except ImportError:
        print('Please use slicer kernel before using this method')

    if show_ct:
        slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", ct_name)
        ct_node = slicer.util.getNode(ct_name)

        ct_node.SetOrigin(list(np.array(my_plan.ct.ct_dict['origin_xyz_mm']) * np.array(
            [-1, -1, 1])))  # convert to RAS origin. -ve in x and y direction
        ct_node.SetSpacing(my_plan.ct.ct_dict['resolution_xyz_mm'])
        ct_node.SetIJKToRASDirections(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))  # set itk to RAS direction
        slicer.util.updateVolumeFromArray(ct_node, my_plan.ct.ct_dict['ct_hu_3d'][0])
        slicer.util.setSliceViewerLayers(background=ct_node)

    # Another way to push simple itk image
    # import sitkUtils
    # ct = sitk.GetImageFromArray(my_plan.ct['ct_hu_3d'][0])
    # ct.SetOrigin(my_plan.ct['origin_xyz_mm'])
    # ct.SetSpacing(my_plan.ct['resolution_xyz_mm'])
    # sitkUtils.PushVolumeToSlicer(ct, targetNode='ct')

    # plot dose
    if show_dose:
        if dose_1d is not None:
            dose_arr = my_plan.inf_matrix.dose_1d_to_3d(dose_1d=dose_1d)
        else:
            dose_1d = sol['inf_matrix'].A * sol['optimal_intensity'] * my_plan.get_num_of_fractions()
            dose_arr = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
        slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", dose_name)
        dose_node = slicer.util.getNode(dose_name)
        dose_node.SetOrigin(list(np.array(my_plan.ct.ct_dict['origin_xyz_mm']) * np.array([-1, -1, 1])))
        dose_node.SetSpacing(my_plan.ct.ct_dict['resolution_xyz_mm'])
        dose_node.SetIJKToRASDirections(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
        slicer.util.updateVolumeFromArray(dose_node, dose_arr)
        slicer.util.setSliceViewerLayers(foreground=dose_node)
        slicer.util.setSliceViewerLayers(foregroundOpacity=0.4)
        # dose_node.GetVolumeDisplayNode.

    # plot structures
    if show_structs:
        structure_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", struct_set_name)
        for i, struct_name in enumerate(my_plan.structures.structures_dict['name']):
            print('Importing struct_name : ' + struct_name)
            slicer.util.addVolumeFromArray(my_plan.structures.structures_dict['structure_mask_3d'][i] * (i + 1),
                                           name=struct_name, nodeClassName="vtkMRMLLabelMapVolumeNode")
            lmap = slicer.util.getNode(struct_name)
            lmap.SetOrigin(list(np.array(my_plan.ct.ct_dict['origin_xyz_mm']) * np.array([-1, -1, 1])))
            lmap.SetSpacing(my_plan.ct.ct_dict['resolution_xyz_mm'])
            lmap.SetIJKToRASDirections(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lmap, structure_node)
            slicer.mrmlScene.RemoveNode(lmap)
            segId = structure_node.GetSegmentation().GetNthSegment(i)
            segId.SetName(struct_name)
