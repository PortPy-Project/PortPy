import scipy.io as spio
import cvxpy as cp
import numpy as np
import mosek
import time


def get_voxels(myPlan, org):
    for i in range(len(myPlan['structures']['Names'])):
        if myPlan['structures']['Names'][i] == org:
            vox = myPlan['structures']['optimizationVoxIndices'][i]

    return vox


def getSmoothnessMatrix(beamReq):
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


def run_imrt_optimization_cvx(myPlan):
    t = time.time()

    infMatrix = myPlan['infMatrixSparse']
    clinicalConstraints = myPlan['clinicalCriteria']['constraints']
    pres = myPlan['clinicalCriteria']['presPerFraction_Gy']
    numFractions = myPlan['clinicalCriteria']['numOfFraction']
    [X, Y] = getSmoothnessMatrix(myPlan['beams'])



    # Construct the problem.
    w = cp.Variable(infMatrix.shape[1], pos=True)
    dO = cp.Variable(len(get_voxels(myPlan, 'PTV')), pos=True)
    dU = cp.Variable(len(get_voxels(myPlan, 'PTV')), pos=True)
    # Form objective.
    print('Objective Start')
    obj = []

    ##Step 1 objective

    ##obj.append((1/sum(pars['points'][(getVoxels(myPlan,'PTV'))-1, 3]))*cp.sum_squares(cp.multiply(cp.sqrt(pars['points'][(getVoxels(myPlan,'PTV'))-1, 3]), infMatrix[getVoxels(myPlan,'PTV')-1, :] @ w + wMean*pars['alpha']*pars['delta'][getVoxels(myPlan,'PTV')-1] - pars['presPerFraction'])))
    obj.append(10000 * (1 / len(get_voxels(myPlan, 'PTV'))) * (cp.sum_squares(dO) + 10 * cp.sum_squares(dU)))

    ##Smoothing objective function

    obj.append(1000*(0.6*cp.sum_squares(X @ w) + 0.4*cp.sum_squares(Y @ w)))

    print('Objective done')
    print('Constraints Start')
    constraints = []
    # constraints += [wMean == cp.sum(w)/w.shape[0]]
    for i in range(len(clinicalConstraints)):
        if 'maxHardConstraint_Gy' in clinicalConstraints[i]:
            if clinicalConstraints[i]['maxHardConstraint_Gy'] is not None:
                org = clinicalConstraints[i]['structNames']
                if org != 'GTV':
                    constraints += [infMatrix[get_voxels(myPlan, org) - 1, :] @ w <= clinicalConstraints[i]['maxHardConstraint_Gy'] / numFractions]
        if 'meanHardConstraint_Gy' in clinicalConstraints[i]:
            if clinicalConstraints[i]['meanHardConstraint_Gy'] is not None:
                org = clinicalConstraints[i]['structNames']
                constraints += [(1 / len(get_voxels(myPlan, org))) * (cp.sum(infMatrix[get_voxels(myPlan, org) - 1, :] @ w)) <= clinicalConstraints[i]['meanHardConstraint_Gy'] / numFractions]

    ##Step 1 and 2 constraint
    constraints += [infMatrix[get_voxels(myPlan, 'PTV') - 1, :] @ w <= pres + dO]
    constraints += [infMatrix[get_voxels(myPlan, 'PTV') - 1, :] @ w >= pres - dU]

    ##Smoothness Constraint
    for b in range(len(myPlan['beams']['Index'])):
          startB = myPlan['beams']['firstBeamlet'][b]
          endB = myPlan['beams']['endBeamlet'][b]
          constraints += [0.6*cp.sum_squares(X[startB-1:endB-1, startB-1:endB-1] @ w[startB-1:endB-1]) + 0.4*cp.sum_squares(Y[startB-1:endB-1, startB-1:endB-1] @ w[startB-1:endB-1]) <= 0.5]

    print('Constraints Done')

    prob = cp.Problem(cp.Minimize(sum(obj)), constraints)
    # Defining the constraints
    print('Problem loaded')
    prob.solve(solver=cp.MOSEK, verbose=True)
    print("optimal value with MOSEK:", prob.value)
    elapsed = time.time() - t
    print('Elapsed time {} seconds'.format(elapsed))
    return w.value


# if __name__ == '__main__':
#
#
#     patientFolderPath = 'F:\Research\Data_newformat\Paraspinal\ECHO_PARAS_3$ECHO_20200003'
#     # read all the meta data for the required patient
#     metaData = loadMetaData(patientFolderPath)
#
#     ##create IMRT Plan
#     gantryRtns = [8, 16, 24, 32, 80, 120]
#     collRtns = [0, 0, 0, 90, 0, 90]
#     beamIndices = [46, 131, 36, 121, 26, 66, 151, 56, 141]
#
#     # myPlan = createIMRTPlan(patientFolderPath, gantryRtns=gantryRtns, collRtns=collRtns)
#     myPlan = createIMRTPlan(patientFolderPath, beamIndices=beamIndices)
#
#     w = runIMRTOptimization_CVX(myPlan)
