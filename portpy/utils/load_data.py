import os
import numpy as np
from scipy.sparse import csr_matrix
import h5py


def load_data(meta_data: dict, pat_dir: str, options: dict) -> dict:
    """
    Takes meta_data and the location of the data as inputs and returns the full data.
    The meta_data only includes light-weight data from the .json files (e.g., beam IDs, angles, structure names,..).
    Large numeric data (e.g., influence matrix, voxel coordinates) are stored in .h5 files.


    :param meta_data: meta_data containing light weight data from json file
    :param pat_dir: patient folder directory containing all the data
    :param options: dictionary for deciding whether to load .h5 data.
    e.g. if options['loadInfluenceMatrixFull']=True, it will load full influence matrix
    :return: a dict of data
    """
    meta_data = load_options(options=options, meta_data=meta_data)  # options to load or ignore the data from .h5 file
    meta_data = load_file(meta_data=meta_data, pat_dir=pat_dir)  # recursive function to load data from .h5 files
    return meta_data


def load_options(options: dict = None, meta_data: dict = None):
    """
    options to load or ignore the data from .h5 file
    :param options: dictionary for deciding whether to load .h5 data.
    e.g. if options['loadInfluenceMatrixFull']=True, it will load full influence matrix
    :param meta_data:
    :return:
    """
    if options is None:
        options = dict()
        options['load_inf_matrix_full'] = False
        options['load_inf_matrix_sparse'] = True
    if len(options) != 0:
        if 'load_inf_matrix_full' in options and not options['load_inf_matrix_full']:
            meta_data['beams_dict']['influenceMatrixFull_File'] = [None] * len(
                meta_data['beams_dict']['influenceMatrixFull_File'])
        if 'load_inf_matrix_sparse' in options and not options['load_inf_matrix_sparse']:
            meta_data['beams_dict']['influenceMatrixSparse_File'] = [None] * len(
                meta_data['beams_dict']['influenceMatrixSparse_File'])
    return meta_data


def load_file(meta_data: dict, pat_dir: str):
    """
    This recursive function loads the data from .h5 files and merge them with the meta_data and returns a dictionary
    including all the data (meta_data+actual numeric data)
    :param meta_data: meta_data containing leight weight data from json file
    :param pat_dir: patient folder directory
    :return:
    """
    for key in meta_data.copy():
        item = meta_data[key]
        if type(item) is dict:
            meta_data[key] = load_file(item, pat_dir)
        elif key == 'beamlets':  # added this part to check if there are beamlets since beamlets are list of dictionary
            if type(item[0]) is dict:
                for ls in range(len(item)):
                    load_file(item[ls], pat_dir)
                    # meta_data[key] = ls_data
        elif key.endswith('_File'):
            success = 1
            for i in range(np.size(meta_data[key])):
                dataFolder = pat_dir
                if meta_data[key][i] is not None:
                    if meta_data[key][i].startswith('Beam_'):
                        dataFolder = os.path.join(dataFolder, 'Beams')
                    if type(meta_data[key]) is not list:
                        if meta_data[key].startswith('Beam_'):  # added this for beamlets
                            dataFolder = os.path.join(dataFolder, 'Beams')
                        file_tag = meta_data[key].split('.h5')
                    else:
                        file_tag = meta_data[key][i].split('.h5')
                    filename = os.path.join(dataFolder, file_tag[0] + '.h5')
                    with h5py.File(filename, "r") as f:
                        if file_tag[1] in f:
                            if key[0:-5] == 'optimizationVoxIndices':
                                vox = f[file_tag[1]][:].ravel()
                                meta_data.setdefault(key[0:-5], []).append(vox.astype(int))
                            elif key[0:-5] == 'BEV_2d_structure_mask':
                                orgs = f[file_tag[1]].keys()
                                organ_mask_dict = dict()
                                for j in orgs:
                                    organ_mask_dict[j] = f[file_tag[1]][j][:]
                                #                                     organ_mask_dict['Mask'].append(f[file_tag[1]][j][:].T)
                                meta_data.setdefault(key[0:-5], []).append(organ_mask_dict)
                            elif key[0:-5] == 'BEV_structure_contour_points':
                                orgs = f[file_tag[1]].keys()
                                organ_mask_dict = dict()
                                for j in orgs:
                                    segments = f[file_tag[1]][j].keys()
                                    for seg in segments:
                                        organ_mask_dict.setdefault(j, []).append(f[file_tag[1]][j][seg][:])
                                        # organ_mask_dict[j] = f[file_tag[1]][j][seg][:].T
                                #                                     organ_mask_dict['Mask'].append(f[file_tag[1]][j][:].T)
                                meta_data.setdefault(key[0:-5], []).append(organ_mask_dict)
                            #                                 meta_data.setdefault(key[0:-5], []).append(f[file_tag[1]][j][:].T)
                            else:
                                meta_data.setdefault(key[0:-5], []).append(f[file_tag[1]][:])
                            if key[0:-5] == 'influenceMatrixSparse':
                                infMatrixSparseForBeam = meta_data[key[0:-5]][i]
                                meta_data[key[0:-5]][i] = csr_matrix(
                                    (infMatrixSparseForBeam[:, 2], (infMatrixSparseForBeam[:, 0].astype(int),
                                                                    infMatrixSparseForBeam[:, 1].astype(int))))
                        else:
                            print('Problem reading Data: {}'.format(meta_data[key][i]))
                            success = 0
            if success:
                del meta_data[key]

    return meta_data
