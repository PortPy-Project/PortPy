import scipy.io as spio
import cvxpy as cp
import numpy as np
import mosek
import time

from utils.misc_utils import get_voxels, get_smoothness_matrix

def run_imrt_optimization_cvx(myPlan):
    t = time.time()

    infMatrix = myPlan['infMatrixSparse']
    clinicalConstraints = myPlan['clinicalCriteria']['constraints']
    pres = myPlan['clinicalCriteria']['presPerFraction_Gy']
    numFractions = myPlan['clinicalCriteria']['numOfFraction']
    [X, Y] = get_smoothness_matrix(myPlan['beams'])



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