from utils import *

import numpy as np
import matplotlib.pyplot as plt

def main():
    patientFolderPath = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Paraspinal_Patient_1'
    saveFilePath = r'C:\Users\fua\Pictures\Figures'
    saveFileName = saveFilePath + r'\robust_smooth-paraspinal.jpg'
    
    # Read all the metadata for the required patient.
    metaData = loadMetaData(patientFolderPath)
    
    # Beams for different patient movements +/- (x,y,z) direction.
    # Gantry angles: 220, 240, 260, 100, 120, 140, 160, 180
    beamIndicesNom = np.arange(9)
    beamIndicesMove = {"Xmin3":   np.arange(9,18),
                       "Xplus3":  np.arange(18,27),
                       "Ymin3":   np.arange(27,36),
                       "Yplus3":  np.arange(36,45),
                       "Zmin3":   np.arange(45,54),
                       "Zplus3":  np.arange(54,63)
                      }

    # Optimization parameters.
    # smoothLambda = [0, 0.15, 0.25, 0.5, 1, 10]
    smoothLambda = [0, 0.5]
    
    # Create IMRT plan for nominal scenario.
    myPlanNom = createIMRTPlan(metaData, beamIndices = beamIndicesNom)
    
    print("Solving NNLS nominal problem...")
    beamOptTrue = []
    doseOptTrue = []
    for lam in smoothLambda:
        print("Smoothing weight: {0}".format(lam))
        w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = runNNLSOptimization_CVX(myPlanNom, cutoff = 0, lambda_x = lam, lambda_y = lam, verbose = False)
        beamOptTrue.append(w_smooth)
        doseOptTrue.append(d_true_smooth)
    
    # Form matrices with rows = beamlets intensities/dose voxels, columns = smoothing weights.
    beamOptTrueMat = np.column_stack(beamOptTrue)
    doseOptTrueMat = np.column_stack(doseOptTrue)
    doseOptTrueNorm = np.linalg.norm(doseOptTrueMat, axis = 0)   # ||A^{nom}*x_l||_2 for l = 1,...,len(smoothLambda).

    # Compute deviation of dose in movement scenarios from nominal dose.
    print("Computing robustness error...")
    doseDiffMatNorm = []
    for scenarioName, beamIndices in beamIndicesMove.items():
        # Import dose-influence matrix for movement scenario.
        print("Movement scenario: {0}".format(scenarioName))
        myPlanMove = createIMRTPlan(metaData, beamIndices = beamIndices)
        infMatMove = myPlanMove["infMatrixSparse"]
        
        doseMoveMat = infMatMove @ beamOptTrueMat              # Rows = dose voxels, columns = smoothing weights.
        doseDiffMat = doseMoveMat - doseOptTrueMat             # Column l = A^{s}*x_l - A^{nom}*x_l, where s = movement scenario.
        doseDiffNorm = np.linalg.norm(doseDiffMat, axis = 0)   # ||A^{s}*x_l - A^{nom}*x_l||_2 for l = 1,...,len(smoothLambda).
        doseDiffMatNorm.append(doseDiffNorm)
    
    # Form matrix with rows = scenarios, columns = smoothing weights.
    doseDiffMatNorm = np.row_stack(doseDiffMatNorm)
    doseDiffSum = np.sum(doseDiffMatNorm, axis = 0)
    doseDiffSumNorm = doseDiffSum/doseOptTrueNorm
   
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    plt.plot(smoothLambda, doseDiffSumNorm)
    plt.xlabel("$\lambda$")
    plt.ylabel("$\sum_{s=1}^N ||A^{s}x_l - A^{nom}x_l||_2/\sum_{s=1}^N ||A^{nom}x_l||_2$")
    plt.title("Robustness Error vs. Smoothing Weight")
    plt.show()
    
    fig.savefig(saveFileName, bbox_inches = "tight", dpi = 300)

if __name__ == "__main__":
    main()
