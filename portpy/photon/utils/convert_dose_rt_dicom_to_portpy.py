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

# Read dicom files from dicom directory
from __future__ import annotations
import os
import SimpleITK as sitk
try:
    from pydicom import dcmread
    pydicom_installed = True
except ImportError:
    pydicom_installed = False
import numpy as np
from portpy.photon.ct import CT
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from portpy.photon.plan import Plan


def read_dicom_dose(dir_name: str = None, dose_file_name: str = None):
    if dir_name is not None:
        dicom_names = os.listdir(dir_name)
        for dcm in dicom_names:
            if 'rt_dose' in dcm:
                dose_file_name = os.path.join(dir_name, dcm)

    dose_img = dcmread(dose_file_name)
    # dose_img = sitk.ReadImage([dcm for dcm in dicom_paths])
    arr_dose = dose_img.pixel_array
    rt_dose = arr_dose * dose_img.DoseGridScaling
    rt_dose_itk = sitk.GetImageFromArray(rt_dose)
    if 'Phantom' in dose_file_name:
        # There is issue for phantom patient. CT and dose dicom images have different origin w.r.t eclipse.
        # Eclipse have different origin
        dose_img.ImagePositionPatient[0] = dose_img.ImagePositionPatient[0] - 65.63
    rt_dose_itk.SetOrigin(dose_img.ImagePositionPatient)
    rt_dose_itk.SetSpacing([float(dose_img.PixelSpacing[0]), float(dose_img.PixelSpacing[1]),
                            dose_img.GridFrameOffsetVector[1] - dose_img.GridFrameOffsetVector[0]])
    # return dose_img
    return rt_dose_itk


def read_dicom(in_dir):
    dicom_names = os.listdir(in_dir)
    dicom_paths = []
    for dcm in dicom_names:
        if dcm[:2] == 'CT':
            dicom_paths.append(os.path.join(in_dir, dcm))

    img_positions = []
    for dcm in dicom_paths:
        ds = dcmread(dcm)
        img_positions.append(ds.ImagePositionPatient[2])

    indexes = np.argsort(np.asarray(img_positions))
    dicom_names = list(np.asarray(dicom_paths)[indexes])

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    img = reader.Execute()

    return img


def resample(img, ref_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetReferenceImage(ref_image)
    img = resampler.Execute(img)

    return img


def get_ct_image(ct: CT):
    ct_arr = ct.ct_dict['ct_hu_3d'][0]
    ct_image = sitk.GetImageFromArray(ct_arr)
    ct_image.SetOrigin(ct.ct_dict['origin_xyz_mm'])
    ct_image.SetSpacing(ct.ct_dict['resolution_xyz_mm'])
    ct_image.SetDirection(ct.ct_dict['direction'])
    return ct_image


def convert_dose_rt_dicom_to_portpy(my_plan: Plan = None, ct: CT = None, dir_name: str = None, dose_file_name: str = None):
    if not pydicom_installed:
        raise ImportError(
            "Pydicom. To use this function, please install it with `pip install portpy[pydicom]`."
        )
    dicom_dose_image = read_dicom_dose(dir_name=dir_name, dose_file_name=dose_file_name)
    if ct is None:
        ct = my_plan.ct
    ct_image = get_ct_image(ct)
    dose_3d_resampled_to_orgCT = resample(dicom_dose_image, ct_image)
    return sitk.GetArrayFromImage(dose_3d_resampled_to_orgCT)
