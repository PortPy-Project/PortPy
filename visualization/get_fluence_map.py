import numpy as np

def get_fluence_map(self):
#Generate the beamlet maps from w

    wMaps = []

    beamReq = self.beams.beams_dict
    for b in range(len(beamReq['ID'])):
        maps = beamReq['beamlet_idx_2dgrid'][b]
        numRows = np.size(maps, 0)
        numCols = np.size(maps, 1)
        wMaps.append(np.zeros((numRows, numCols)))
        for r in range(numRows):
            for c in range(numCols):
                if maps[r, c] >= 0:
                    curr = maps[r, c]
                    wMaps[b][r, c] = self.optimal_intensity[curr]

    return wMaps