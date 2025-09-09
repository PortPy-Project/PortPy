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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from portpy.photon.plan import Plan
try:
    from pydicom import dcmread
    from pydicom import Dataset, Sequence
    pydicom_installed = True
except ImportError:
    pydicom_installed = False


def write_rt_plan_imrt(my_plan: Plan, leaf_sequencing: dict, out_rt_plan_file: str, in_rt_plan_file: str):
    """

    Write output from leaf sequencing to dicom rt plan

    :param my_plan: object of class plan
    :param leaf_sequencing: dictionary containing leaf positions and shapes
    :param in_rt_plan_file: file location of input rt plan containing imrt beams
    :param out_rt_plan_file: new rt plan which can be imported in TPS

    """
    # # get positions. PortPy have same jaw settings for all beams
    # top_left_x_mm = my_plan.beams.beams_dict['jaw_position'][0]['top_left_x_mm']
    # bottom_right_x_mm = my_plan.beams.beams_dict['jaw_position'][0]['bottom_right_x_mm']
    # bottom_right_y_mm = my_plan.beams.beams_dict['jaw_position'][0]['bottom_right_y_mm']
    # top_left_y_mm = my_plan.beams.beams_dict['jaw_position'][0]['top_left_y_mm']

    if not pydicom_installed:
        raise ImportError(
            "Pydicom. To use this function, please install it with `pip install portpy[pydicom]`."
        )

    # read rt plan file using pydicom
    ds = dcmread(in_rt_plan_file)
    for b in range(len(ds.BeamSequence)):
        gantry_angle = int(ds.BeamSequence[b].ControlPointSequence[0].GantryAngle)
        meterset_weight = 0
        if gantry_angle in leaf_sequencing:
            leaf_postions = leaf_sequencing[gantry_angle]['leaf_postions']
            # get positions. PortPy have same jaw settings for all beams
            idx = my_plan.beams.beams_dict['gantry_angle'].index(gantry_angle)
            top_left_x_mm = my_plan.beams.beams_dict['jaw_position'][idx]['top_left_x_mm']
            bottom_right_x_mm = my_plan.beams.beams_dict['jaw_position'][idx]['bottom_right_x_mm']
            bottom_right_y_mm = my_plan.beams.beams_dict['jaw_position'][idx]['bottom_right_y_mm']
            top_left_y_mm = my_plan.beams.beams_dict['jaw_position'][idx]['top_left_y_mm']
            del ds.BeamSequence[b].ControlPointSequence[1:]  # delete all the control point after 1st control point
            ds.BeamSequence[b].NumberOfControlPoints = len(leaf_postions) + 1
            for cp, shape in enumerate(leaf_postions):
                ds.BeamSequence[b].ControlPointSequence.append(Dataset())
                ds.BeamSequence[b].ControlPointSequence[cp+1].ControlPointIndex = str(cp + 1)
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence = Sequence()
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence.append(Dataset())

                # X jaw.[X1 X2] -ve of the X1 we see in eclipse
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence[0].RTBeamLimitingDeviceType = 'ASYMX'
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence[0].LeafJawPositions = [top_left_x_mm, bottom_right_x_mm]

                # Y jaw.[Y1 Y2] -ve of the Y1 we see in eclipse
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence.append(Dataset())
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence[1].RTBeamLimitingDeviceType = 'ASYMY'
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence[1].LeafJawPositions = [bottom_right_y_mm, top_left_y_mm]

                # set leafs
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence.append(Dataset())
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence[2].RTBeamLimitingDeviceType = 'MLCX'
                ds.BeamSequence[b].ControlPointSequence[cp+1].BeamLimitingDevicePositionSequence[2].LeafJawPositions = shape

                meterset_weight = meterset_weight + leaf_sequencing[gantry_angle]['MU'][cp] / sum(leaf_sequencing[gantry_angle]['MU'])
                ds.BeamSequence[b].ControlPointSequence[cp+1].CumulativeMetersetWeight = round(meterset_weight, 4)

    ds.save_as(out_rt_plan_file)
