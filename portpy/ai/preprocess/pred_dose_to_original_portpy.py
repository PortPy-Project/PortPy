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
import argparse


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


def process_case(in_dir, case):
    # ct = get_dataset(in_dir, case, '_CT.nrrd')
    # dose = get_dataset(in_dir, case, '_dose_resampled.nrrd')
    oar = get_dataset(in_dir, case, '_RTSTRUCTS.nrrd')
    ptv = get_dataset(in_dir, case, '_PTV.nrrd')

    oar[np.where(ptv == 1)] = 6

    start, end = get_crop_settings(oar)

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


cases = ['Lung_Patient_9', 'Lung_Patient_10']
# Adding optional argument
# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--planName", required=False)
parser.add_argument("--in_dir", required=False, help="Enter input dir having patient folders with their dicoms")
# parser.add_argument("--gt_dir", required=False, help="Enter gt dir having patient predicted dose in downsampled resolution")
# parser.add_argument("--out_dir", required=False, help="Enter out dir where dose is saved in portpy resolution")
args, _ = parser.parse_known_args()
in_dir = args.in_dir
# gt_dir = args.gt_dir
# out_dir = args.out_dir
planName = args.planName


planName = 'portpy_test_1'
in_dir = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\PortPy\data'
gt_dir = r'../results/{}/test_latest/npz_images'.format(planName)
out_dir = r'../results/{}/test_latest/pred_dose'.format(planName)

labels = {
    'cord': 1,
    'esophagus': 2,
    'heart': 3,
    'lung_l': 4,
    'lung_r': 5,
    'ptv': 1
}  # PTV will be stored separately as its extent is not mutually exclusive with other anatomies

for idx, case in enumerate(cases):
    print('Processing case {}: {} of {} ...'.format(case, idx + 1, len(cases)))
    # read dicom CT and write it in out_dir
    data = pp.DataExplorer(data_dir=in_dir)
    data.patient_id = case
    meta_data = data.load_metadata()
    # Load ct and structure set for the above patient using CT and Structures class
    ct = pp.CT(data)
    ct_arr = ct.ct_dict['ct_hu_3d'][0]
    structs = pp.Structures(data)
    ct_img = get_ct_image(ct)
    # find crop setting for the patient
    oars = ['cord', 'esophagus', 'heart', 'lung_l', 'lung_r', 'ptv']
    target_oars = dict.fromkeys(oars, -1)  # Will store index of target OAR contours from dicom dataset

    oar_mask = np.zeros(ct_arr.shape, np.uint8)
    ptv_mask = np.zeros_like(oar_mask)

    for k, v in target_oars.items():
        # anatomy_mask = np.zeros_like(oar_mask)
        ind = structs.structures_dict['name'].index(k.upper())
        anatomy_mask = structs.structures_dict['structure_mask_3d'][ind]

        if k == 'ptv':
            ptv_mask[np.where(anatomy_mask > 0)] = labels[k]
        else:
            oar_mask[np.where(anatomy_mask > 0)] = labels[k]

    oar_mask[np.where(ptv_mask == 1)] = 6

    # start, end = get_crop_settings(oar_mask)
    start, end = get_crop_settings_calc_box(ct, meta_data=meta_data)
    # read predicted dose in down sampled resolution
    filename = os.path.join(gt_dir, case + '_CT2DOSE.nrrd')
    pred_dose = sitk.ReadImage(filename)
    pred_dose = sitk.GetArrayFromImage(pred_dose)

    ct_arr_cropped = ct_arr[start[0]:end[0] + 1, start[1]:end[1], start[2]:end[2]]

    pred_dose_to_ct_crop = resample_dose(pred_dose, ct_arr_cropped)  # First get pred dose to cropped ct dimensions
    pred_dose = attach_slices(pred_dose_to_ct_crop, ct_img, start, end)  # attach empty slices
    # pred_dose_resampled = resample(pred_dose, gt_dose)  # resample the undo-crop to original dose

    filename = os.path.join(out_dir, case + '_pred_dose_original_resolution.nrrd')
    sitk.WriteImage(pred_dose, filename)

    # visualizing predicted dvh using portpy
    beams = pp.Beams(data)
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)
    planner_dose_3d = pp.convert_dose_rt_dicom_to_portpy(ct=ct,
                                                         dose_file_name=os.path.join(in_dir, case, 'rt_dose.dcm'))

    planner_dose_1d = inf_matrix.dose_3d_to_1d(dose_3d=planner_dose_3d)
    pred_dose_1d = inf_matrix.dose_3d_to_1d(dose_3d=sitk.GetArrayFromImage(pred_dose))

    import matplotlib.pyplot as plt

    # load clinical criteria from the config files for which plan to be optimized
    protocol_name = 'Lung_2Gy_30Fx'
    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)

    my_plan = pp.Plan(ct, structs, beams, inf_matrix, clinical_criteria)

    struct_names = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=planner_dose_1d, struct_names=struct_names, style='solid', ax=ax, norm_flag=True)
    ax = pp.Visualization.plot_dvh(my_plan, dose_1d=pred_dose_1d, struct_names=struct_names, style='dotted', ax=ax, norm_flag=True)
    ax.set_title('- Planner .. Predicted')
    plt.show()
print('Done')
