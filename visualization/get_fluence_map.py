import numpy as np
def get_fluence_map(myPlan, w):
#Generate the beamlet maps from w

    wMaps = []

    beamReq = myPlan['beams']
    for b in range(len(beamReq['Index'])):
        maps = beamReq['beamEyeViewBeamletMap'][b]
        numRows = np.size(maps, 0)
        numCols = np.size(maps, 1)
        wMaps.append(np.zeros((numRows, numCols)))
        for r in range(numRows):
            for c in range(numCols):
                if maps[r, c] > 0:
                    curr = maps[r, c]
                    wMaps[b][r, c] = w[curr-1]

    return wMaps