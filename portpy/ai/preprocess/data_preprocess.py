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

import sys

sys.path.append(r'..\..')
import argparse
import os
import portpy.photon as pp
import SimpleITK as sitk
import numpy as np
import torch
from skimage.transform import resize
import traceback


# remember to modify this function in future if you want to add more sites or use protocol_name to determine prescription dose
def get_site_config(site: str):
    if site.lower() == 'lung':
        disease_struct_names = ['cord', 'esophagus', 'heart', 'lungs_not_gtv', 'ptv']
        default_prescription_gy = 60
    elif site.lower() == 'prostate':
        disease_struct_names = ['bladder', 'rectum', 'femur_l', 'femur_r', 'ptv']
        default_prescription_gy = 70.2
    else:
        raise ValueError("site must be either 'lung' or 'prostate'")

    labels = {name: i + 1 for i, name in enumerate(disease_struct_names[:-1])}
    labels['ptv'] = 1

    return disease_struct_names, labels, default_prescription_gy

def get_structure_masks(structs: pp.Structures, ct_shape, labels, disease_struct_names, case=''):
    struct_names = structs.structures_dict['name']
    struct_names_upper = [s.upper() for s in struct_names]

    oar_mask = np.zeros(ct_shape, np.uint8)
    ptv_mask = np.zeros_like(oar_mask)

    for k in disease_struct_names:
        if k == 'lungs_not_gtv':
            if 'LUNGS_NOT_GTV' in struct_names_upper:
                ind = struct_names_upper.index('LUNGS_NOT_GTV')
                anatomy_mask = structs.structures_dict['structure_mask_3d'][ind]
            else: # create lungs not gtv for some patients
                anatomy_mask = np.zeros(ct_shape, dtype=bool)

                if 'LUNG_L' in struct_names_upper:
                    ind_l = struct_names_upper.index('LUNG_L')
                    anatomy_mask = np.logical_or(
                        anatomy_mask,
                        structs.structures_dict['structure_mask_3d'][ind_l] > 0
                    )

                if 'LUNG_R' in struct_names_upper:
                    ind_r = struct_names_upper.index('LUNG_R')
                    anatomy_mask = np.logical_or(
                        anatomy_mask,
                        structs.structures_dict['structure_mask_3d'][ind_r] > 0
                    )

                if 'GTV' in struct_names_upper:
                    ind_gtv = struct_names_upper.index('GTV')
                    gtv_mask = structs.structures_dict['structure_mask_3d'][ind_gtv] > 0
                    anatomy_mask = np.logical_and(anatomy_mask, ~gtv_mask)

            oar_mask[np.where(anatomy_mask > 0)] = labels[k]

        elif k.upper() in struct_names_upper:
            ind = struct_names_upper.index(k.upper())
            anatomy_mask = structs.structures_dict['structure_mask_3d'][ind]

            if k == 'ptv':
                ptv_mask[np.where(anatomy_mask > 0)] = labels[k]
            else:
                oar_mask[np.where(anatomy_mask > 0)] = labels[k]
        else:
            print(f'Structure {k} not found for case {case}. Leaving it as zero mask.')

    return oar_mask, ptv_mask

# def get_structure_masks(structs: pp.Structures, ct_shape, labels, disease_struct_names, case=''):
#     struct_names = structs.structures_dict['name']
#     struct_names_upper = [s.upper() for s in struct_names]
#
#     oar_mask = np.zeros(ct_shape, np.uint8)
#     ptv_mask = np.zeros_like(oar_mask)
#
#     for k in disease_struct_names:
#         if k.upper() in struct_names_upper:
#             ind = struct_names_upper.index(k.upper())
#             anatomy_mask = structs.structures_dict['structure_mask_3d'][ind]
#
#             if k == 'ptv':
#                 ptv_mask[np.where(anatomy_mask > 0)] = labels[k]
#             else:
#                 oar_mask[np.where(anatomy_mask > 0)] = labels[k]
#         else:
#             print(f'Structure {k} not found for case {case}. Leaving it as zero mask.')
#
#     return oar_mask, ptv_mask


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


def write_image(img_arr, out_dir, case, suffix, ref_ct):
    img_itk = sitk.GetImageFromArray(img_arr)
    img_itk.SetOrigin(ref_ct.GetOrigin())
    img_itk.SetSpacing(ref_ct.GetSpacing())
    img_itk.SetDirection(ref_ct.GetDirection())
    filename = os.path.join(out_dir, case + suffix)
    sitk.WriteImage(img_itk, filename)


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


def get_torch_tensor(npy_tensor, device):
    out = torch.from_numpy(npy_tensor)
    out.to(device)

    return out


def get_dvh(dose, oar, ptv):
    # Compute and return the dvh for all 6 OAR structures
    device = torch.device('cuda:0')
    dose = get_torch_tensor(dose, device)
    oar = get_torch_tensor(oar, device).long()
    oar = torch.nn.functional.one_hot(oar, 6)[..., 1:]  # Remove BG
    oar = oar.permute(3, 0, 1, 2).to(torch.float)
    ptv = get_torch_tensor(ptv, device).long().unsqueeze(dim=0)
    ptv = ptv.to(torch.float)
    oar = torch.cat((oar, ptv), axis=0)

    vols = torch.sum(oar, axis=(1, 2, 3))
    n_bins = 351
    hist = torch.zeros((n_bins, 6)).to(device)
    bins = torch.linspace(0, 70, n_bins)
    bin_w = bins[1] - bins[0]

    for i in range(bins.shape[0]):
        diff = torch.sigmoid((dose - bins[i]) / bin_w)
        diff = torch.cat(6 * [diff.unsqueeze(axis=0)]) * oar
        num = torch.sum(diff, axis=(1, 2, 3))
        hist[i] = (num / vols)

    hist_numpy = hist.cpu().numpy()
    bins_np = bins.cpu().numpy()

    return hist_numpy, bins_np


def process_case(ct_portpy, meta_data, ct, oar, ptv, beamlet, out_dir, case,
                 prescription_gy, dose = None, body=None):
    # oar_copy = oar.copy()
    # oar_copy[np.where(ptv == 1)] = 6

    start, end = get_crop_settings_calc_box(ct_portpy, meta_data=meta_data)

    ct = crop_resize_img(ct, start, end, is_mask=False)
    oar = crop_resize_img(oar, start, end, is_mask=True)
    ptv = crop_resize_img(ptv, start, end, is_mask=True)
    if dose is not None:
        dose = crop_resize_img(dose, start, end, is_mask=False)
    if body is not None:
        body = crop_resize_img(body, start, end, is_mask=True)
    beamlet = crop_resize_img(beamlet, start, end, is_mask=False)

    beamlet[np.where(ptv == 1)] = prescription_gy

    # correct version with bug fix
    if dose is not None:
        # num_ptv = np.sum(ptv)
        # ptv_sum = np.sum(dose * ptv)
        #
        # if num_ptv > 0 and ptv_sum > 0:
        #     ptv_mean = ptv_sum / num_ptv
        #     scale_factor = prescription_gy / ptv_mean
        #     dose = dose * scale_factor

        dose = np.clip(dose, a_min=0, a_max=1.2*prescription_gy).astype(np.float32)

    ct = np.clip(ct, a_min=-1000, a_max=3071)
    ct = (ct + 1000) / 4071
    ct = ct.astype(np.float32)

    beamlet = beamlet.astype(np.float32)

    save_dict = {
        "CT": ct,
        "DOSE": dose,
        "OAR": oar,
        "PTV": ptv,
        "BEAM": beamlet,
        "PRESCRIPTION_GY": np.float32(prescription_gy),
    }
    if body is not None:
        save_dict["BODY"] = body.astype(np.uint8)

    filename = os.path.join(out_dir, case)
    np.savez(filename, **save_dict)


def data_preprocess(in_dir, out_dir, site: str='lung', protocol_name: str = None, beam_ids = None, technique_name: str = 'imrt'):
    """
    in_dir: raw data input directory
    out_dir: processed data out directory
    site: anatomical site. Default is lung. In future, can add more sites and modify code accordingly.
    protocol_name: protocol name to determine prescription dose. Default is None, which means it will use the prescription dose from metadata. In future, can modify code to use protocol_name to determine prescription dose.
    dose_clip_max_gy: maximum dose in Gy to clip the dose array. Default is None, which means it will use 1.2 times the prescription dose as the maximum dose for clipping. Can modify this in future to use a fixed value or a value based on protocol_name.
    """
    if not os.path.isabs(in_dir):
        base_dir = os.getcwd()  # get current running directory
        in_dir = os.path.join(base_dir, in_dir)
    if not os.path.isabs(out_dir):
        base_dir = os.getcwd()  # get current running directory
        out_dir = os.path.join(base_dir, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cases = os.listdir(in_dir)

    disease_struct_names, labels, default_prescription_gy = get_site_config(site)
    for idx, case in enumerate(cases):
        # if dose file does not exist in raw data dir, skip it
        dose_file_path = os.path.join(in_dir, case, 'DicomFiles', 'rt_dose_echo_{}.dcm'.format(technique_name.lower()))
        if not os.path.exists(dose_file_path):
            print('Dose file does not exist for case {}. Skipping...'.format(case))
            continue
        # process all the nrrd files
        try:

            if site.lower() in case.lower():
                if case == 'Lung_Phantom_Patient_1':
                    # These patient doesnt include some structures. Modify code in future
                    continue
                print('Processing case {}: {} of {} ...'.format(case, idx + 1, len(cases)))
                # read dicom CT and write it in out_dir
                data = pp.DataExplorer(data_dir=in_dir)
                data.patient_id = case
                meta_data = data.load_metadata()
                if protocol_name is not None:
                    clinical_criteria = pp.ClinicalCriteria(data, protocol_name=protocol_name)
                    prescription_gy = clinical_criteria.get_prescription()
                else:
                    prescription_gy = default_prescription_gy
                # Load ct and structure set for the above patient using CT and Structures class
                ct = pp.CT(data)
                ct_arr = ct.ct_dict['ct_hu_3d'][0]
                structs = pp.Structures(data)
                beams = pp.Beams(data, beam_ids=beam_ids)
                inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)
                beams_1d = inf_matrix.A * np.ones((inf_matrix.A.shape[1]))
                beams_3d = inf_matrix.dose_1d_to_3d(dose_1d=beams_1d)
                beams_3d = beams_3d.astype('float16')

                # normalize beams_3d. Don't forget add these lines
                beams_3d = ((beams_3d - np.amin(beams_3d)) / (np.amax(beams_3d) - np.amin(beams_3d))) * 1.2 * prescription_gy
                planner_dose_3d = pp.convert_dose_rt_dicom_to_portpy(ct=ct,
                                                                     dose_file_name=os.path.join(in_dir, case, 'DicomFiles', 'rt_dose_echo_{}.dcm'.format(technique_name.lower())))

                oar_mask, ptv_mask = get_structure_masks(
                    structs=structs,
                    ct_shape=ct_arr.shape,
                    labels=labels,
                    disease_struct_names=disease_struct_names,
                    case=case
                )

                # skip patient only if ptv is missing
                if np.sum(ptv_mask) == 0:
                    print(f'PTV not found for case {case}. Skipping this case.')
                    continue
                body_mask = structs.get_structure_mask_3d('BODY').astype(np.uint8)
                process_case(
                    ct_portpy=ct,
                    meta_data=meta_data,
                    ct=ct_arr,
                    dose=planner_dose_3d,
                    oar=oar_mask,
                    ptv=ptv_mask,
                    beamlet=beams_3d,
                    out_dir=out_dir,
                    case=case,
                    prescription_gy=prescription_gy,
                    body=body_mask)

        except:
            print('Processing of case {} failed'.format(case))
            print(str(traceback.format_exc()))
            pass


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("--in_dir", required=False, help="Enter input dir having patient folders with their dicoms")
    parser.add_argument("--out_dir", required=False, help="Enter out dir having patient folders with their dicoms")
    args, _ = parser.parse_known_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    data_preprocess(in_dir=in_dir, out_dir=out_dir)
