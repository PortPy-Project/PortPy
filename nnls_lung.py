from utils import *

import numpy as np
import matplotlib.pyplot as plt

def main():
    patientFolderPath = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Lung_Patient_2'
    fileFigurePath = r'C:\Users\fua\Pictures\Figures'
    fileFigureName = fileFigurePath + r'\robust_smooth-lung_2.jpg'
    
    # Read all the meta data for the required patient
    metaData = loadMetaData(patientFolderPath)
    # metaData['beams']['influenceMatrixFull_File'] = [None]*len(metaData['beams']['influenceMatrixFull_File'])   # Skip loading full dose influence matrix.

    # gantryRtns = [8, 16, 24, 32, 80, 120]
    # collRtns = [0, 0, 0, 90, 0, 90]
    # beamIndices = [46,131,36,121,26,66,151,56,141]
    beamIndices = [10, 20, 30, 40]
    smoothLambda = [0, 0.15, 0.25, 0.5, 1, 10]
    # smoothLambda = [0, 0.15]
    # cutoff = 0.1
    cutoff = 0.01
    vol_perc = 0.9

    # Create IMRT Plan
    print("Creating IMRT plan...")
    myPlan = createIMRTPlan(metaData, beamIndices = beamIndices)
    pres = myPlan['clinicalCriteria']['presPerFraction_Gy']
    i_ptv = getVoxels(myPlan, "PTV") - 1

    # print("Solving NNLS cutoff problem...")
    # w, obj, rob = runNNLSOptimization_CVX(myPlan, cutoff = 0.025, verbose = False)

    # print("\nSolving NNLS cutoff problem with smoothing...")
    # w_smooth, obj_smooth, rob_smooth = runNNLSOptimization_CVX(myPlan, cutoff = 0.025, lambda_x = 0.6, lambda_y = 0.4, verbose = False)
    
    print("Solving NNLS cutoff problem...")
    doseTrueDiff = []
    for lam in smoothLambda:
        print("Smoothing weight: {0}".format(lam))
        # w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = runNNLSOptimization_CVX(myPlan, cutoff = cutoff, lambda_x = lam, lambda_y = lam, sparse = True, verbose = False)
        w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = runNNLSOptimization_CVX(myPlan, cutoff = cutoff, lambda_x = lam, lambda_y = lam, sparse = False, verbose = False)
        
        # Scale dose vectors so V(90%) = p, i.e., 90% of PTV receives 100% of prescribed dose.
        d_true_smooth[i_ptv] = scaleDose(d_true_smooth[i_ptv], pres, vol_perc)
        d_cut_smooth[i_ptv] = scaleDose(d_cut_smooth[i_ptv], pres, vol_perc)
        
        rob_smooth = np.linalg.norm(d_cut_smooth - d_true_smooth)/np.linalg.norm(d_true_smooth)
        doseTrueDiff.append(rob_smooth)
    
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    plt.plot(smoothLambda, doseTrueDiff)
    # plt.xlim(left = 0)
    # plt.ylim(bottom = 0)
    plt.xlabel("$\lambda$")
    # plt.ylabel("$||A^{cutoff}x - A^{true}x||_2$")
    plt.ylabel("$||A^{cutoff}x - A^{true}x||_2/||A^{true}x||_2$")
    plt.title("Robustness Error vs. Smoothing Weight (Lung Patient 2)")
    plt.show()
    
    fig.savefig(fileFigureName, bbox_inches = "tight", dpi = 300)

if __name__ == "__main__":
    main()
