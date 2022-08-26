import numpy as np

# Scale dose vector so V(vol_perc%) = p, i.e., vol_perc% of voxels receive 100% of prescribed dose p.
def scale_dose(d, p, vol_perc):
    d_perc = np.percentile(d, 1 - vol_perc)
    scale = p / d_perc
    return scale*d

def get_voxels(myPlan, org):
    for i in range(len(myPlan['structures']['Names'])):
        if myPlan['structures']['Names'][i] == org:
            vox = myPlan['structures']['optimizationVoxIndices'][i]
    return vox

def get_smoothness_matrix(beamReq):
    sRow = np.zeros((beamReq['endBeamlet'][-1], beamReq['endBeamlet'][-1]), dtype=int)
    sCol= np.zeros((beamReq['endBeamlet'][-1], beamReq['endBeamlet'][-1]), dtype=int)
    for b in range(len(beamReq['Index'])):
        map = beamReq['beamEyeViewBeamletMap'][b]

        rowsNoRepeat = [0]
        for i in range(1, np.size(map, 0)):
            if (map[i,:] != map[rowsNoRepeat[-1], :]).any():
                rowsNoRepeat.append(i)
        colsNoRepeat = [0]
        for j in range(1, np.size(map, 1)):
            if (map[:, j] != map[:, colsNoRepeat[-1]]).any():
                colsNoRepeat.append(j)
        map = map[np.ix_(np.asarray(rowsNoRepeat), np.asarray(colsNoRepeat))]
        for r in range(np.size(map, 0)):
            startCol = 0
            endCol = np.size(map, 1) - 2
            while (map[r, startCol] == 0) and (startCol <= endCol):
                startCol = startCol + 1
            while ((map[r, endCol] == 0) and (startCol <= endCol)):
                endCol = endCol - 1

            for c in range(startCol, endCol+1):
                ind = map[r, c]
                RN = map[r, c + 1]
                if ind * RN > 0:
                    sRow[ind-1, ind-1] = int(1)
                    sRow[ind-1, RN-1] = int(-1)

        for c in range(np.size(map, 1)):
            startRow = 0
            endRow = np.size(map, 0) - 2
            while (map[startRow, c] == 0) and (startRow <= endRow):
                startRow = startRow + 1
            while (map[endRow, c] == 0) and (startRow <= endRow):
                endRow = endRow - 1
            for r in range(startRow, endRow+1):
                ind = map[r, c]
                DN = map[r+1, c]
                if ind * DN > 0:
                    sCol[ind-1, ind-1] = int(1)
                    sCol[ind-1, DN-1] = int(-1)
    return sRow, sCol
