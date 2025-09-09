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
    from portpy.photon.beam import Beams
try:
    from pydicom import dcmread
    from pydicom import Dataset, Sequence
    pydicom_installed = True
except ImportError:
    pydicom_installed = False
import numpy as np


def create_cp(my_plan, beam_id, cp_index, meterset_weight, leaf_positions):
    beams: Beams = my_plan.beams
    jaw_positions = beams.get_jaw_positions(beam_id)
    top_left_x_mm = jaw_positions['top_left_x_mm']
    bottom_right_x_mm = jaw_positions['bottom_right_x_mm']
    bottom_right_y_mm = jaw_positions['bottom_right_y_mm']
    top_left_y_mm = jaw_positions['top_left_y_mm']
    energy_mv = beams.beams_dict['energy_MV'][0]
    iso_center = beams.get_iso_center(beam_id)


    control_point = Dataset()
    control_point.ControlPointIndex = str(cp_index)
    control_point.NominalBeamEnergy = energy_mv
    control_point.DoseRateSet = 600  # hard coded
    BeamLimitingDevicePositionSequence = Sequence()

    beam_limiting_dataset_x = Dataset()
    # X jaw.[X1 X2] -ve of the X1 we see in eclipse
    beam_limiting_dataset_x.RTBeamLimitingDeviceType = 'ASYMX'
    beam_limiting_dataset_x.LeafJawPositions = [top_left_x_mm, bottom_right_x_mm+my_plan.beams.get_beamlet_width()] # Temporary bug fix
    BeamLimitingDevicePositionSequence.append(beam_limiting_dataset_x)

    # Y jaw.[Y1 Y2] -ve of the Y1 we see in eclipse
    beam_limiting_dataset_y = Dataset()
    beam_limiting_dataset_y.RTBeamLimitingDeviceType = 'ASYMY'
    beam_limiting_dataset_y.LeafJawPositions = [bottom_right_y_mm - my_plan.beams.get_beamlet_height(), top_left_y_mm]
    BeamLimitingDevicePositionSequence.append(beam_limiting_dataset_y)

    beam_limiting_dataset_leaf_positions = Dataset()
    beam_limiting_dataset_leaf_positions.RTBeamLimitingDeviceType = 'MLCX'
    beam_limiting_dataset_leaf_positions.LeafJawPositions = leaf_positions
    BeamLimitingDevicePositionSequence.append(beam_limiting_dataset_leaf_positions)

    control_point.BeamLimitingDevicePositionSequence = BeamLimitingDevicePositionSequence
    control_point.GantryAngle = beams.get_gantry_angle(beam_id)
    control_point.GantryRotationDirection = "CW"
    control_point.CumulativeMetersetWeight = meterset_weight

    # Additional attributes
    if cp_index == 0:
        control_point.BeamLimitingDeviceAngle = 0
        control_point.BeamLimitingDeviceRotationDirection = 'NONE'
        control_point.PatientSupportAngle = 0
        control_point.PatientSupportRotationDirection = 'NONE'
        control_point.TableTopEccentricAngle = 0.0
        control_point.TableTopEccentricRotationDirection = "NONE"
        control_point.TableTopVerticalPosition = 0.0
        control_point.TableTopLongitudinalPosition = 1000.0
        control_point.TableTopLateralPosition = 0.0
        control_point.IsocenterPosition = [iso_center['x_mm'], iso_center['y_mm'], iso_center['z_mm']]
        control_point.TableTopPitchAngle = 0.0
        control_point.TableTopPitchRotationDirection = "NONE"
        control_point.TableTopRollAngle = 0.0
        control_point.TableTopRollRotationDirection = "NONE"
    return control_point

def create_beam(my_plan, arc_id, beam_number):
    ind = [i for i in range(len(my_plan.arcs.arcs_dict['arcs'])) if
           my_plan.arcs.arcs_dict['arcs'][i]['arc_id'] == arc_id][0]
    leaf_y = [-200, -190, -180, -170, -160, -150, -140, -130, -120, -110, -100, -95, -90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    beam = Dataset()
    beam.Manufacturer = "Varian Medical Systems"
    beam.ManufacturerModelName = "TDS"
    beam.DeviceSerialNumber = "1003"

    # Primary Fluence Mode Sequence
    primary_fluence_mode = Dataset()
    primary_fluence_mode.FluenceMode = "STANDARD"
    beam.PrimaryFluenceModeSequence = [primary_fluence_mode]

    beam.TreatmentMachineName = "445HET1"
    beam.PrimaryDosimeterUnit = "MU"
    beam.SourceAxisDistance = my_plan.beams.beams_dict['SAD_mm'][0]

    # Beam Limiting Device Sequence
    beam_limiting_device_sequence = Sequence()

    # ASYMX
    asxmx = Dataset()
    asxmx.RTBeamLimitingDeviceType = "ASYMX"
    asxmx.NumberOfLeafJawPairs = "1"
    beam_limiting_device_sequence.append(asxmx)

    # ASYMY
    asxmy = Dataset()
    asxmy.RTBeamLimitingDeviceType = "ASYMY"
    asxmy.NumberOfLeafJawPairs = "1"
    beam_limiting_device_sequence.append(asxmy)

    # MLCX
    mlcx = Dataset()
    mlcx.RTBeamLimitingDeviceType = "MLCX"
    mlcx.SourceToBeamLimitingDeviceDistance = "509.0"  # Hard coded. modify it
    mlcx.NumberOfLeafJawPairs = "60"
    mlcx.LeafPositionBoundaries = leaf_y
    beam_limiting_device_sequence.append(mlcx)

    beam.BeamLimitingDeviceSequence = beam_limiting_device_sequence

    beam.BeamNumber = str(beam_number)
    beam.BeamName = f"Field {arc_id}"
    beam.BeamType = "DYNAMIC"
    beam.RadiationType = "PHOTON"
    beam.TreatmentDeliveryType = "TREATMENT"
    beam.NumberOfWedges = "0"
    beam.NumberOfCompensators = "0"
    beam.NumberOfBoli = "0"
    beam.NumberOfBlocks = "0"
    beam.FinalCumulativeMetersetWeight = np.round(1.0, 4)
    beam.NumberOfControlPoints = str(len(my_plan.arcs.arcs_dict['arcs'][ind]['vmat_opt']))
    beam.ReferencedPatientSetupNumber = beam_number
    return beam

def edit_fraction_group_sequence(my_plan, ds):
    ds.FractionGroupSequence[0].NumberOfBeams = str(len(my_plan.arcs.arcs_dict['arcs']))
    from pydicom.uid import generate_uid

    reference_dose_uid = generate_uid()
    # Define Referenced Beam Sequence
    referenced_beam_sequence = Sequence()
    patient_setup_sequence = Sequence()
    for i in range(len(my_plan.arcs.arcs_dict['arcs'])):
        # Add Referenced Beam attributes
        referenced_beam = Dataset()
        referenced_beam.ReferencedDoseReferenceUID = reference_dose_uid
        referenced_beam.ReferencedBeamNumber = str(i+1)

        # Add Referenced Beam to sequence
        referenced_beam_sequence.append(referenced_beam)
        # Create Patient Setup attributes
        patient_setup = Dataset()
        patient_setup.PatientPosition = "HFS"
        patient_setup.PatientSetupNumber = str(i+1)
        patient_setup.SetupTechnique = "ISOCENTRIC"

        # Add Patient Setup to sequence
        patient_setup_sequence.append(patient_setup)

    # Add Patient Setup Sequence to RT Plan
    ds.PatientSetupSequence = patient_setup_sequence
    ds.FractionGroupSequence[0].ReferencedBeamSequence = referenced_beam_sequence

    # refer new dose
    dose_reference_sequence = Sequence()
    # Create Dose Reference attributes
    dose_reference = Dataset()
    dose_reference.DoseReferenceNumber = "1"
    dose_reference.DoseReferenceUID = reference_dose_uid
    # Add Dose Reference to sequence
    dose_reference_sequence.append(dose_reference)
    # Add Dose Reference Sequence to RT Plan
    ds.DoseReferenceSequence = dose_reference_sequence

    return ds
def write_rt_plan_vmat(my_plan: Plan, out_rt_plan_file: str, in_rt_plan_file: str):
    """

    Write output from leaf sequencing to dicom rt plan

    :param my_plan: object of class plan
    :param in_rt_plan_file: file location of input rt plan containing imrt beams
    :param out_rt_plan_file: new rt plan which can be imported in TPS

    """
    if not pydicom_installed:
        raise ImportError(
            "Pydicom. To use this function, please install it with `pip install portpy[pydicom]`."
        )
    # read rt plan file using pydicom
    ds = dcmread(in_rt_plan_file)
    del ds.BeamSequence
    arcs = my_plan.arcs
    for a, arc in enumerate(arcs.arcs_dict['arcs']):
        for b, beam in enumerate(arc['vmat_opt']):
            beam['field_size'] = beam['best_leaf_position_in_cm'][:, 1]*10 - beam['best_leaf_position_in_cm'][:, 0]*10
            beam_id = beam['beam_id']
            beamlets = my_plan.beams.get_beamlets(beam_id=beam_id)
            min_origin_x = np.min(beamlets['position_x_mm'])
            beam['bank_b'] = min_origin_x + beam['best_leaf_position_in_cm'][:, 0]*10 - my_plan.beams.get_finest_beamlet_width() / 2
            beam['bank_a'] = min_origin_x + beam['best_leaf_position_in_cm'][:, 1]*10 + my_plan.beams.get_finest_beamlet_width() / 2

    # Write bank positions to text file
    # the leaf pairs between max_leaf_pair and min_leaf_pair must satisfy the dynamic min leaf gap constraint
    # min gap between leaf pairs should be at least 0.5 mm
    total_lp = 60  # Temporary. Tell Hai to add it to metadata
    ds.BeamSequence = Sequence()
    # ds = edit_fraction_group_sequence(my_plan, ds)
    for a, arc in enumerate(arcs.arcs_dict['arcs']):
        arc_id = arc['arc_id']
        beam_dataset = create_beam(my_plan=my_plan, arc_id=arc_id, beam_number=a+1)
        mu = [arc['vmat_opt'][i]['best_beam_weight']*100 for i in range(len(arc['vmat_opt']))] # multiply it by 100 to match eclipse mu/deg
        mu[0] = 0  # make first beam 0 mu
        total_mu = sum(mu)
        meterset_weight = 0
        control_point_sequence = Sequence()
        for b, beam in enumerate(arc['vmat_opt']):
            beam_id = beam['beam_id']
            leaf_pos = np.empty((total_lp, 2), dtype='object')
            stop = int(beam['start_leaf_pair']) - 1  # We are doing it in reverse way since rt plan have reverse index list
            for r in range(beam['num_rows']):
                start = stop - len(beam['MLC_leaf_idx'][0][0])
                if beam['field_size'][r] < 0.5:
                    leaf_pos[start:stop, 1] = [0]
                else:
                    leaf_pos[start:stop, 0] = [beam['bank_b'][r]]
                    leaf_pos[start:stop, 1] = [beam['bank_a'][r]]
                stop = stop - len(beam['MLC_leaf_idx'][0][0])

            leaf_pos[:, 0][leaf_pos[:, 0] == None] = 0.5
            leaf_pos[:, 1][leaf_pos[:, 1] == None] = 0.5
            a = leaf_pos.flatten('F')
            string_list = [str(item) for item in a.tolist()]
            if b > 0:
                meterset_weight += beam['best_beam_weight'] / total_mu
            control_point = create_cp(my_plan, beam_id, b, meterset_weight, string_list)
            control_point_sequence.append(control_point)
        beam_dataset.ControlPointSequence = control_point_sequence
        ds.BeamSequence.append(beam_dataset)
    ds.save_as(out_rt_plan_file)
    print('Dicom rt plan file created..')
