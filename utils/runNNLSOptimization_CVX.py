import scipy.io as spio
import scipy.sparse as spsparse
import cvxpy as cp
import numpy as np
import mosek
import time
from utils import *

def runNNLSOptimization_CVX(myPlan, cutoff = 0, lambda_x = 0, lambda_y = 0, sparse = True, verbose = True):
    if cutoff < 0 or cutoff > 1:
        raise ValueError("cutoff must be a scalar in [0,1]")
    if lambda_x < 0:
        raise ValueError("lambda_x must be a nonnegative scalar")
    if lambda_y < 0:
        raise ValueError("lambda_y must be a nonnegative scalar")
    
    t = time.time()
    
    infMatrix = myPlan['infMatrixSparse'] if sparse else myPlan['infMatrixFull']
    clinicalConstraints = myPlan['clinicalCriteria']['constraints']
    pres = myPlan['clinicalCriteria']['presPerFraction_Gy']
    numFractions = myPlan['clinicalCriteria']['numOfFraction']
    
    n_voxels, n_beams = infMatrix.shape
    i_ptv = getVoxels(myPlan, "PTV") - 1
    i_oar = np.setdiff1d(np.arange(n_voxels), i_ptv)
    n_ptv = len(i_ptv)
    n_oar = n_voxels - n_ptv
	
	# Zero out small values in influence matrix.
    if cutoff == 0:
        infMatrixCut = infMatrix
    else:
        infMatrixMax = infMatrix.max()
        infMatrixCut = infMatrix.copy()
        if spsparse.issparse(infMatrix):
            infMatrixCut.data[infMatrixCut.data <= cutoff*infMatrixMax] = 0
        else:
            infMatrixCut[infMatrixCut <= cutoff*infMatrixMax] = 0
        print('Truncated influence matrix to {0} of max value'.format(cutoff))
	
    # Construct the problem.
    w = cp.Variable(n_beams, nonneg = True)
    
    # Dose penalty function.
    obj = (10000/n_ptv)*cp.sum_squares(infMatrix[i_ptv,:] @ w - pres)
    # obj_oar = (10000/n_oar)*cp.sum_squares(infMatrix[i_oar,:] @ w)
    
    # Smoothing regularization term.
    if not (lambda_x == 0 and lambda_y == 0):
        [X, Y] = getSmoothnessMatrix(myPlan['beams'])
        reg = 1000*(lambda_x*cp.sum_squares(X @ w) + lambda_y*cp.sum_squares(Y @ w))
        obj = obj + reg
    print("Objective done")
    
    constraints = []
    # constraints += [wMean == cp.sum(w)/w.shape[0]]
    for i in range(len(clinicalConstraints)):
        if 'maxHardConstraint_Gy' in clinicalConstraints[i] and clinicalConstraints[i]['maxHardConstraint_Gy'] is not None:
            org = clinicalConstraints[i]['structNames']
            constraints += [infMatrix[getVoxels(myPlan, org)-1, :] @ w <= clinicalConstraints[i]['maxHardConstraint_Gy']]
        if 'meanHardConstraint_Gy' in clinicalConstraints[i] and clinicalConstraints[i]['meanHardConstraint_Gy'] is not None:
            org = clinicalConstraints[i]['structNames']
            constraints += [(1/len(getVoxels(myPlan, org)))*(cp.sum(infMatrix[getVoxels(myPlan, org)-1, :] @ w)) <= clinicalConstraints[i]['meanHardConstraint_Gy']]
    print("Constraints done")
    
    prob = cp.Problem(cp.Minimize(obj))
    print("Problem loaded")
    
    prob.solve(solver = cp.MOSEK, verbose = verbose)
    if prob.status not in cp.settings.SOLUTION_PRESENT:
        raise RuntimeError("Solver failed with status {0}".format(prob.status))
    w_opt = w.value
    obj_opt = prob.value
    
    elapsed = time.time() - t
    
    print("Optimal value with MOSEK:", obj_opt)
    print('Elapsed time {} seconds'.format(elapsed))
    
    d_true = infMatrix @ w_opt
    d_cut = infMatrixCut @ w_opt
    # d_diff = np.linalg.norm(d_cut - d_true)
    d_diff = np.linalg.norm(d_cut - d_true)/np.linalg.norm(d_true)
    print("Relative difference from true dose:", d_diff)
    
    return w_opt, obj_opt, d_true, d_cut
