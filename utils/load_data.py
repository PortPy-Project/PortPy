from . import *
import os

import numpy as np
from scipy.sparse import csr_matrix
import h5py


def load_data(myData, folderPath):
    # fn = myData.keys()
    for key in myData.copy():
        item = myData[key]
        if type(item) is dict:
            myData[key] = load_data(item, folderPath)
        elif key.endswith('_File'):
            success = 1
            for i in range(np.size(myData[key])):
                dataFolder = folderPath
                if myData[key][i] is not None:
                    if myData[key][i].startswith('Beam_'):
                        dataFolder = os.path.join(dataFolder, 'Beams')
                    if type(myData[key]) is not list:
                        file_tag = myData[key].split('.h5')
                    else:
                        file_tag = myData[key][i].split('.h5')
                    filename = os.path.join(dataFolder, file_tag[0] + '.h5')
                    with h5py.File(filename, "r") as f:
                        if file_tag[1] in f:
                            if key[0:-5] == 'optimizationVoxIndices':
                                vox = f[file_tag[1]][:].T.ravel()
                                myData.setdefault(key[0:-5], []).append(vox.astype(int))
                            elif key[0:-5] == 'BEV_2d_structure_mask':
                                orgs = f[file_tag[1]].keys()
                                organ_mask_dict = dict()
                                for j in orgs:
                                    organ_mask_dict[j] = f[file_tag[1]][j][:].T
#                                     organ_mask_dict['Mask'].append(f[file_tag[1]][j][:].T)
                                myData.setdefault(key[0:-5], []).append(organ_mask_dict)
                            elif key[0:-5] == 'BEV_structure_contour_points':
                                orgs = f[file_tag[1]].keys()
                                organ_mask_dict = dict()
                                for j in orgs:
                                    segments = f[file_tag[1]][j].keys()
                                    for seg in segments:
                                        organ_mask_dict.setdefault(j, []).append(f[file_tag[1]][j][seg][:].T)
                                        # organ_mask_dict[j] = f[file_tag[1]][j][seg][:].T
                                #                                     organ_mask_dict['Mask'].append(f[file_tag[1]][j][:].T)
                                myData.setdefault(key[0:-5], []).append(organ_mask_dict)
#                                 myData.setdefault(key[0:-5], []).append(f[file_tag[1]][j][:].T)
                            else:
                                myData.setdefault(key[0:-5], []).append(f[file_tag[1]][:].T)
                            if key[0:-5] == 'influenceMatrixSparse' or key[0:-5] == 'influenceMatrixFull':
                                infMatrixSparseForBeam = myData[key[0:-5]][i]
                                myData[key[0:-5]][i] = csr_matrix((infMatrixSparseForBeam[:, 2], (infMatrixSparseForBeam[:, 0].astype(int),
                                             infMatrixSparseForBeam[:, 1].astype(int))))
                        else:
                            print('Problem reading Data: {}'.format(myData[key][i]))
                            success = 0
            if success:
                del myData[key]

    return myData