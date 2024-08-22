import os
import SimpleITK as sitk
from pydicom import dcmread
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.spatial import cKDTree
import h5py


def read_dicom(in_dir, case):
    dicom_names = os.listdir(os.path.join(in_dir, case))
    dicom_paths = []
    for dcm in dicom_names:
        if dcm[:2] == 'CT':
            dicom_paths.append(os.path.join(in_dir, case, dcm))

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


def create_ct_dose_vox_map_zyx(ct: sitk.Image, points_xyz: np.ndarray, uniform: bool = False, output_folder=None,
                               num_points=None) -> np.ndarray:
    ct_dose_map_zyx = np.ones_like(sitk.GetArrayFromImage(ct), dtype=int) * int(-1)
    if not uniform:
        for point_num, row in enumerate(points_xyz):
            curr_indx = ct.TransformPhysicalPointToIndex(row)  # X,Y,Z
            ct_dose_map_zyx[curr_indx[::-1]] = point_num  # zyx
            if point_num == num_points-1:     # Temp due to Bug #TODO
                break

        mask = np.where(ct_dose_map_zyx > -1)
        z_max, z_min = np.amax(mask[0]), np.amin(mask[0])
        y_max, y_min = np.amax(mask[1]), np.amin(mask[1])
        x_max, x_min = np.amax(mask[2]), np.amin(mask[2])
        calc_box = ct_dose_map_zyx[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]

        dose_vox_ind = np.where(calc_box > -1)
        no_dose_vox_ind = np.where(calc_box == -1)
        a, nearest_ind = cKDTree(np.asarray(dose_vox_ind).T).query(np.asarray(no_dose_vox_ind).T)
        calc_box[no_dose_vox_ind] = calc_box[dose_vox_ind[0][nearest_ind], dose_vox_ind[1][nearest_ind], dose_vox_ind[2][nearest_ind]]
        ct_dose_map_zyx[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1] = calc_box
        return ct_dose_map_zyx
    else:
        fname = os.path.join(output_folder, 'OptimizationVoxels_MetaData.json')
        # Opening JSON file
        f = open(fname)
        voxels_metadata = json.load(f)
        dose_res = voxels_metadata['dose_voxel_resolution_xyz_mm']
        vox_res_int = [round(b / m) for b, m in zip(dose_res, ct.GetSpacing())]
        for point_num, row in enumerate(points_xyz):
            # row is dose voxel point. to get ct point for it, go to corner point in dose voxel and add ct resolution/2
            curr_pt = (row[0] - (dose_res[0] / 2) + (ct.GetSpacing()[0] / 2), row[1] - (dose_res[1] / 2) + (ct.GetSpacing()[1] / 2), row[2] - (dose_res[2] / 2) + (ct.GetSpacing()[2] / 2))
            curr_indx = ct.TransformPhysicalPointToIndex(curr_pt)  # X,Y,Z

            ct_dose_map_zyx[curr_indx[2]:(curr_indx[2] + vox_res_int[2]), curr_indx[1]:(curr_indx[1] + vox_res_int[1]),
            curr_indx[0]:(curr_indx[0] + vox_res_int[0])] = point_num  # zyx
        return ct_dose_map_zyx

def write_image(img_arr, out_dir, case, suffix, ref_ct):
    img_itk = sitk.GetImageFromArray(img_arr)
    img_itk.SetOrigin(ref_ct.GetOrigin())
    img_itk.SetSpacing(ref_ct.GetSpacing())
    img_itk.SetDirection(ref_ct.GetDirection())
    filename = os.path.join(out_dir, case + suffix)
    sitk.WriteImage(img_itk, filename)


def load_json(file_name):
    f = open(file_name)
    json_data = json.load(f)
    f.close()
    return json_data


def create_ct_dose_voxel_map(output_folder, num_points):
    # output_folder = r'\\pisiz3echo\ECHO\Prostate\Test\outputs\ECHO_PROST_1$ECHO_20200009\HP_PyTest_GJ_IMRT'
    print('starting python code..')
    if os.path.exists(os.path.join(output_folder, 'CT')):
        ct = read_dicom(output_folder, 'CT')
        ct_zyx = sitk.GetArrayFromImage(ct)  # zyx
        # sitk.WriteImage(ct, os.path.join(output_folder, 'CT.nrrd'))

        with h5py.File(os.path.join(output_folder, 'CT_Data.h5'), 'w') as hf:
            hf.create_dataset("ct_hu_3d", data=ct_zyx, chunks=True, compression='gzip', compression_opts=9)
    else:
        opt_metadata = load_json(os.path.join(output_folder, 'OptimizationVoxels_MetaData.json'))
        ct = sitk.Image(opt_metadata['ct_size_xyz'], sitk.sitkInt32)
        ct.SetOrigin(opt_metadata['ct_origin_xyz_mm'])
        ct.SetSpacing(opt_metadata['ct_voxel_resolution_xyz_mm'])
        ct.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    # creating voxel map

    filename = os.path.join(output_folder, 'OptimizationVoxels_Data.h5')

    with h5py.File(filename, "r") as f:
        # if 'ct_to_dose_voxel_map' in f:
        #     return
        if 'voxel_coordinate_XYZ_mm' in f:
            voxel_coordinate_XYZ_mm = f['voxel_coordinate_XYZ_mm'][:]

    # create ct to dose voxel map
    print('Creating ct to dose voxel map..')
    ct_dose_map_zyx = create_ct_dose_vox_map_zyx(ct, voxel_coordinate_XYZ_mm, uniform=False, output_folder=output_folder, num_points=num_points)

    # consider only inside body dose voxels
    with h5py.File(os.path.join(output_folder, 'StructureSet_Data.h5'), 'r') as f:
        patient_surface_name = [s for s in f.keys() if 'Patient S' in s]
        if patient_surface_name:
            body = f[patient_surface_name[0]][:]
        else:
            body = f['BODY'][:]
    ct_dose_map_zyx = ((ct_dose_map_zyx + 1)*body)-1

    # write ct to doze voxel map in optimization voxel data.h5
    with h5py.File(
            os.path.join(output_folder, 'OptimizationVoxels_Data.h5'), 'a') as hf:
        if 'ct_to_dose_voxel_map' in hf.keys():
            del hf['ct_to_dose_voxel_map']
        hf.create_dataset("ct_to_dose_voxel_map", data=ct_dose_map_zyx, chunks=True, compression='gzip',
                          compression_opts=9)


if __name__ == "__main__":
    # create an hdf5 file for Ct and CT to voxel map
    output_folder = r'data'
    create_ct_dose_voxel_map(output_folder)