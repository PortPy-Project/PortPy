import numpy as np
from utils.loadData import loadData
from utils.infMatrixConcatenate import infMatrixConcatenate


def createIMRTPlan(metaData, beamIndices=None):

    myPlan = metaData.copy()
    del myPlan['beams']
    beamReq = dict()
    inds = []
    for i in range(len(beamIndices)):
        if beamIndices[i] in metaData['beams']['Index']:
                    ind = np.where(np.array(metaData['beams']['Index']) == beamIndices[i])
                    ind = ind[0][0]
                    inds.append(ind)
                    for key in metaData['beams']:
                        beamReq.setdefault(key, []).append(metaData['beams'][key][ind])
    myPlan['beams'] = beamReq
    if len(inds) < len(beamIndices):
        print('some indices are not available')
    myPlan = loadData(myPlan, myPlan['patientFolderPath'])
    myPlan = infMatrixConcatenate(myPlan)

    return myPlan
# if __name__ == "__main__":
#     patientFolderPath = r'F:\\Research\\Data_newformat\\Paraspinal\\ECHO_PARAS_3$ECHO_20200003'
#     gantryRtns = [12, 20, 40]
#     collRtns = [0, 0, 90]
#     myPlan = createIMRTPlan(patientFolderPath, gantryRtns=gantryRtns, collRtns=collRtns)
