import numpy as np
from scipy.sparse import hstack


def infMatrixConcatenate(myPlan):
    if 'influenceMatrixSparse' in myPlan['beams']:
        # myPlan['infMatrixSparse'] = []
        for i in range(np.size(myPlan['beams']['Index'])):
            map_i = myPlan['beams']['beamEyeViewBeamletMap'][i]
            if i == 0:
                myPlan['infMatrixSparse'] = myPlan['beams']['influenceMatrixSparse'][i]
            else:
                myPlan['infMatrixSparse'] = hstack([myPlan['infMatrixSparse'], myPlan['beams']['influenceMatrixSparse'][i]], format='csr')
            logicalMap = map_i > 0
            a = np.sort(map_i.flatten())
            a = np.unique(a)
            secondMin = a[1]
            # secondMin = a[1]
            if i != 0:
                standardMap = np.int32(map_i) - secondMin * np.int32(logicalMap) + np.int32(logicalMap)
                map = standardMap + np.amax(myPlan['beams']['beamEyeViewBeamletMap'][i - 1]) * np.int32(logicalMap)
            else:
                map = np.int32(map_i) - secondMin * np.int32(logicalMap) + np.int32(logicalMap)

            myPlan['beams']['beamEyeViewBeamletMap'][i] = map
            standInd = np.unique(np.sort(map.flatten()))
            myPlan['beams'].setdefault('firstBeamlet', []).append(standInd[1])
            myPlan['beams'].setdefault('endBeamlet', []).append(np.amax(map))
        del myPlan['beams']['influenceMatrixSparse']
    if 'influenceMatrixFull' in myPlan['beams']:
        myPlan['infMatrixFull'] = []
        for i in range(np.size(myPlan['beams'])):
            map_i = myPlan['beams']['beamEyeViewBeamletMap'][i]
            if i == 0:
                myPlan['infMatrixFull'] = myPlan['beams']['influenceMatrixFull'][i]
            else:
                myPlan['infMatrixFull'] = hstack([myPlan['infMatrixFull'], myPlan['beams']['influenceMatrixFull'][i]], format='csr')
            logicalMap = map_i > 0
            a = np.sort(map_i.flatten())
            a = np.unique(a)
            secondMin = a[1]
            # secondMin = a[1]
            if i != 0:
                standardMap = np.int32(map_i) - secondMin * np.int32(logicalMap) + np.int32(logicalMap)
                map = standardMap + np.amax(myPlan['beams']['beamEyeViewBeamletMap'][i - 1]) * np.int32(logicalMap)
            else:
                map = np.int32(map_i) - secondMin * np.int32(logicalMap) + np.int32(logicalMap)

            myPlan['beams']['beamEyeViewBeamletMap'][i] = map
            standInd = np.unique(np.sort(map.flatten()))
            myPlan['beams'].setdefault('firstBeamlet', []).append(standInd[1])
            myPlan['beams'].setdefault('endBeamlet', []).append(np.amax(map))
        del myPlan['beams']['influenceMatrixFull']
    return myPlan