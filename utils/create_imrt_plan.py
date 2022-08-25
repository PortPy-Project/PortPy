import numpy as np
from utils.load_data import load_data
from utils.inf_matrix_concatenate import inf_matrix_concatenate


def create_imrt_plan(metaData, options=None, beam_indices=None):
    if len(options) != 0:
        if 'loadInfluenceMatrixFull' in options and not options['loadInfluenceMatrixFull']:
            metaData['beams']['influenceMatrixFull_File'] = [None] * len(metaData['beams']['influenceMatrixFull_File'])
        if 'loadInfluenceMatrixSparse' in options and not options['loadInfluenceMatrixSparse']:
            metaData['beams']['influenceMatrixSparse_File'] = [None] * len(metaData['beams']['influenceMatrixSparse_File'])
        if 'loadBeamEyeViewStructureMask' in options and not options['loadBeamEyeViewStructureMask']:
            metaData['beams']['beamEyeViewStructureMask_File'] = [None] * len(metaData['beams']['beamEyeViewStructureMask_File'])
    my_plan = metaData.copy()
    del my_plan['beams']
    beamReq = dict()
    inds = []
    for i in range(len(beam_indices)):
        if beam_indices[i] in metaData['beams']['Index']:
                    ind = np.where(np.array(metaData['beams']['Index']) == beam_indices[i])
                    ind = ind[0][0]
                    inds.append(ind)
                    for key in metaData['beams']:
                        beamReq.setdefault(key, []).append(metaData['beams'][key][ind])
    my_plan['beams'] = beamReq
    if len(inds) < len(beam_indices):
        print('some indices are not available')
    my_plan = load_data(my_plan, my_plan['patientFolderPath'])
    my_plan = inf_matrix_concatenate(my_plan)

    return my_plan
# if __name__ == "__main__":
#     patientFolderPath = r'F:\\Research\\Data_newformat\\Paraspinal\\ECHO_PARAS_3$ECHO_20200003'
#     gantryRtns = [12, 20, 40]
#     collRtns = [0, 0, 90]
#     myPlan = createIMRTPlan(patientFolderPath, gantryRtns=gantryRtns, collRtns=collRtns)
