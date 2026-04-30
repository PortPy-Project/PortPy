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

import SimpleITK as sitk
import numpy as np
import os
from skimage.transform import resize
import portpy.photon as pp
# import argparse
from portpy.ai.preprocess.data_preprocess import get_site_config, get_structure_masks, process_case
from portpy.ai.test import test
from portpy.ai.infer_runner import InferenceRunner


def get_dataset(in_dir, case, suffix):
    filename = os.path.join(in_dir, case + suffix)
    img = None
    if os.path.exists(filename):
        img = sitk.ReadImage(filename)
        img = sitk.GetArrayFromImage(img)

    return img


def get_ct_image(ct: pp.CT):
    ct_arr = ct.ct_dict['ct_hu_3d'][0]
    ct_image = sitk.GetImageFromArray(ct_arr)
    ct_image.SetOrigin(ct.ct_dict['origin_xyz_mm'])
    ct_image.SetSpacing(ct.ct_dict['resolution_xyz_mm'])
    ct_image.SetDirection(ct.ct_dict['direction'])

    return ct_image


def resample(img, ref_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetReferenceImage(ref_image)
    img = resampler.Execute(img)

    return img


def resample_dose(dose, ref_dose):
    dims = ref_dose.shape
    dose = np.moveaxis(dose, 0, -1)  # Channels last
    expected_shape = (dims[1], dims[2], dims[0])
    dose = resize(dose, expected_shape, order=1, preserve_range=True, anti_aliasing=False)
    dose = np.moveaxis(dose, -1, 0)

    return dose


def crop_arr(img, start, end, is_mask=False):
    # Crop to setting given by start/end coordinates list, assuming depth,height,width
    img_arr = sitk.GetArrayFromImage(img)
    img_cropped = img_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]
    img_cropped = sitk.GetImageFromArray(img_cropped)
    img_cropped.SetOrigin(img.GetOrigin())
    img_cropped.SetDirection(img.GetDirection())
    img_cropped.SetSpacing(img.GetSpacing())
    # img_cropped = np.moveaxis(img_cropped, 0, -1)  # Slices last
    #
    # order = 0
    # if is_mask is False:
    #     order = 1
    # img_resized = resize(img_cropped, (128, 128, 128), order=order, preserve_range=True, anti_aliasing=False).astype(np.float32)
    # if is_mask is True:
    #     img_resized = img_resized.astype(np.uint8)
    #
    # img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

    return img_cropped


def crop_resize_img(img, start, end, is_mask=False):
    # Crop to setting given by start/end coordinates list, assuming depth,height,width

    img_cropped = img[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]
    img_cropped = np.moveaxis(img_cropped, 0, -1)  # Slices last

    order = 0
    if is_mask is False:
        order = 1
    img_resized = resize(img_cropped, (128, 128, 128), order=order, preserve_range=True, anti_aliasing=False).astype(
        np.float32)
    if is_mask is True:
        img_resized = img_resized.astype(np.uint8)

    img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

    return img_resized

def get_crop_settings_calc_box(ct: pp.CT, meta_data):
    cal_box_xyz_start = meta_data['opt_voxels']['cal_box_xyz_start']
    cal_box_xyz_end = meta_data['opt_voxels']['cal_box_xyz_end']
    ct_img = get_ct_image(ct)
    start_xyz = ct_img.TransformPhysicalPointToIndex(cal_box_xyz_start)  # X,Y,Z
    end_xyz = ct_img.TransformPhysicalPointToIndex(cal_box_xyz_end)  # X,Y,Z
    start_zyx = [start_xyz[2], start_xyz[1], start_xyz[0]]
    end_zyx = [end_xyz[2], end_xyz[1], end_xyz[0]]
    return start_zyx, end_zyx


def get_crop_settings(oar):
    # Use to get crop settings
    # Don't use cord or eso as they spread through more slices
    # If total number of slices is less than 128 then don't crop at all
    # Use start and end index from presence of any anatomy or ptv
    # If that totals more than 128 slices then leave as is.
    # If that totals less than 128 slices then add slices before and after to make total slices to 128

    oar1 = oar.copy()
    oar1[np.where(oar == 1)] = 0
    oar1[np.where(oar == 2)] = 0

    # For 2D cropping just do center cropping 256x256
    center = [0, oar.shape[1] // 2, oar1.shape[2] // 2]
    start = [0, center[1] - 150, center[2] - 150]
    end = [0, center[1] + 150, center[2] + 150]

    depth = oar1.shape[0]
    if depth < 128:
        start[0] = 0
        end[0] = depth

        return start, end

    first_slice = -1
    last_slice = -1
    for i in range(depth):
        frame = oar1[i]
        if np.any(frame):
            first_slice = i
            break
    for i in range(depth - 1, -1, -1):
        frame = oar1[i]
        if np.any(frame):
            last_slice = i
            break

    expanse = last_slice - first_slice + 1
    if expanse >= 128:
        start[0] = first_slice
        end[0] = last_slice

        return start, end

    # print('Get\'s here')
    slices_needed = 128 - expanse
    end_slices = slices_needed // 2
    beg_slices = slices_needed - end_slices

    room_available = depth - expanse
    end_room_available = depth - last_slice - 1
    beg_room_available = first_slice

    leftover_beg = beg_room_available - beg_slices
    if leftover_beg < 0:
        end_slices += np.abs(leftover_beg)
        first_slice = 0
    else:
        first_slice = first_slice - beg_slices

    leftover_end = end_room_available - end_slices
    if leftover_end < 0:
        first_slice -= np.abs(leftover_end)
        last_slice = depth - 1
    else:
        last_slice = last_slice + end_slices

    if first_slice < 0:
        first_slice = 0

    start[0] = first_slice
    end[0] = last_slice

    return start, end


def attach_slices(pred_dose, ct_img, start, end):
    ct_arr = sitk.GetArrayFromImage(ct_img)
    ref_dose_arr = np.zeros_like(ct_arr, dtype=float)
    # ref_dose_copy = ref_dose
    ref_dose_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]] = pred_dose
    # dose = ref_dose_arr
    # ref_dose = sitk.GetImageFromArray(ref_dose)
    dose = sitk.GetImageFromArray(ref_dose_arr)
    dose.SetOrigin(ct_img.GetOrigin())
    dose.SetDirection(ct_img.GetDirection())
    dose.SetSpacing(ct_img.GetSpacing())

    return dose


def predict_using_model(patient_id, in_dir, out_dir=r'./dataset/infer', model_name='portpy_test_1', checkpoints_dir='../checkpoints', results_dir=r'../results',
                        netG = 'unet_128', site='lung', protocol_name=None, beam_ids=None, infer_runner: InferenceRunner = None):
    """

    :param patient_id: Patient ID for which to run inference. Should match the patient ID in the input directory.
    :param in_dir: input directory containing the DICOM files for the patient. Should be organized in the same way as the input directory for training (i.e. with subdirectories for CT, structures, beams, etc.)
    :param out_dir: output directory to save the preprocessed data for inference. This will be created if it doesn't exist. The preprocessed data will be saved in a subdirectory called 'infer' within this directory.
    :param model_name: model name to use for inference. This should match the name of the model used during training and the name of the subdirectory in the checkpoints directory where the model weights are saved.
    :param checkpoints_dir: directory where the model checkpoints are saved. The function will look for the latest checkpoint in the subdirectory corresponding to the model name.
    :param results_dir: results directory where the predicted dose will be saved. The predicted dose will be saved in a subdirectory corresponding to the model name, under 'test_latest/npz_images'.
    :param netG: network architecture to use for inference. This should match the architecture used during training (e.g. 'unet_128', 'mednext').
    :param site: disease site for which to run inference. This will be used to determine the relevant structures and labels for preprocessing the data. Should match the site used during training (e.g. 'lung', 'prostate').
    :param protocol_name: protocol name to use for inference. This will be used to determine the prescription dose for the case, which will be used to rescale the predicted dose. If not provided, the default prescription dose for the site will be used.
    :param beam_ids: beam IDs to use for inference. This will be used to determine which beams to include in the influence matrix calculation. If not provided, planner beams will be included.
    :param infer_runner: InferenceRunner instance to use for running inference. If not provided, the function will run inference using the test function directly. This allows for flexibility in how inference is run, such as using a custom runner that handles batching or distributed inference.
    :return:
    """
    gt_dir = os.path.join(results_dir, model_name, "test_latest", "npz_images")  # directory to save predicted results
    # directory to save preprocessed data

    # create test directory in out_dir
    out_dir = os.path.join(out_dir, 'infer')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

        # preprocess data
    print('Processing case {}...'.format(patient_id))
    # read dicom CT and write it in out_dir
    data = pp.DataExplorer(data_dir=in_dir)
    data.patient_id = patient_id
    meta_data = data.load_metadata()
    # Load ct and structure set for the above patient using CT and Structures class
    ct = pp.CT(data)
    ct_arr = ct.ct_dict['ct_hu_3d'][0]
    structs = pp.Structures(data)
    ct_img = get_ct_image(ct)

    beams = pp.Beams(data, beam_ids=beam_ids)
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)
    beams_1d = inf_matrix.A * np.ones((inf_matrix.A.shape[1]))
    beams_3d = inf_matrix.dose_1d_to_3d(dose_1d=beams_1d)
    beams_3d = beams_3d.astype('float16')

    disease_struct_names, labels, default_prescription_gy = get_site_config(site)

    if protocol_name is not None:
        clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
        prescription_gy = clinical_criteria.get_prescription()
    else:
        prescription_gy = default_prescription_gy

    # rescale beams_3d between 0 to 72(max dose) 1.2*prescription
    beams_3d = ((beams_3d - np.amin(beams_3d)) / (np.amax(beams_3d) - np.amin(beams_3d))) * 1.2*prescription_gy



    oar_mask, ptv_mask = get_structure_masks(
        structs=structs,
        ct_shape=ct_arr.shape,
        labels=labels,
        disease_struct_names=disease_struct_names,
        case=patient_id
    )

    if np.sum(ptv_mask) == 0:
        raise ValueError(f'PTV not found for case {patient_id}. Cannot run inference.')
    # print('Processing case {}: {} of {} ...'.format(case, idx+1, len(cases)))
    process_case(
        ct_portpy=ct,
        meta_data=meta_data,
        ct=ct_arr,
        oar=oar_mask,
        ptv=ptv_mask,
        beamlet=beams_3d,
        out_dir=out_dir,
        case=patient_id,
        prescription_gy=prescription_gy,
        dose=None
    )

    # get crop settings
    start, end = get_crop_settings_calc_box(ct, meta_data=meta_data)

    # create prediction
    test_file_path = os.path.join(os.path.dirname(__file__), "..")
    # print('Testing script is located at {}'.format(test_file_path))
    # test_file_path = os.path.join(test_file_path, 'test.py')
    # os.system(
    #     'python {} --dataroot {} --netG unet_128 --name {} --checkpoints_dir {} --phase test --model test --eval --input_nc 8 --output_nc 1 --results_dir {} --direction AtoB --dataset_mode single --norm batch'.format(test_file_path, out_dir, model_name, checkpoints_dir, results_dir))

    # test({
    #     "dataroot": out_dir,
    #     "netG": netG,
    #     "name": model_name,
    #     "checkpoints_dir": checkpoints_dir,
    #     "phase": "test",
    #     "model": "test",
    #     "eval": True,
    #     "input_nc": 8,
    #     "output_nc": 1,
    #     "results_dir": results_dir,
    #     "direction": "AtoB",
    #     "dataset_mode": "single",
    #     "norm": "batch",
    # })
    infer_args = {
        "dataroot": out_dir,
        "netG": netG,
        "name": model_name,
        "checkpoints_dir": checkpoints_dir,
        "phase": "test",
        "model": "test",
        "eval": True,
        "input_nc": 8,
        "output_nc": 1,
        "results_dir": results_dir,
        "direction": "AtoB",
        "dataset_mode": "single",
        "norm": "batch",
    }

    if infer_runner is None:
        test(infer_args)
    else:
        infer_runner.run(out_dir)
    # read predicted dose in down sampled resolution
    filename = os.path.join(gt_dir, patient_id + '_CT2DOSE.nrrd')
    pred_dose = sitk.ReadImage(filename)
    pred_dose = sitk.GetArrayFromImage(pred_dose)

    # convert predicted dose to original resolution
    ct_arr_cropped = ct_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]

    pred_dose_to_ct_crop = resample_dose(pred_dose, ct_arr_cropped)  # First get pred dose to cropped ct dimensions
    pred_dose = attach_slices(pred_dose_to_ct_crop, ct_img, start, end)  # attach empty slices
    pred_dose_3d = sitk.GetArrayFromImage(pred_dose)
    #
    # filename = os.path.join(out_dir, case + '_pred_dose_original_resolution.nrrd')
    # sitk.WriteImage(pred_dose, filename)
    return pred_dose_3d
