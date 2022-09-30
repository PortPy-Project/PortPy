import numpy as np
from .visualization import Visualization

def get_fluence_map(self):
#Generate the beamlet maps from w

    wMaps = []

    beamReq = self.beams.beams_dict
    for b in range(len(beamReq['ID'])):
        maps = beamReq[''][b]
        numRows = np.size(maps, 0)
        numCols = np.size(maps, 1)
        wMaps.append(np.zeros((numRows, numCols)))
        for r in range(numRows):
            for c in range(numCols):
                if maps[r, c] > 0:
                    curr = maps[r, c]
                    wMaps[b][r, c] = self.optimal_intensity[curr-1]

    return wMaps