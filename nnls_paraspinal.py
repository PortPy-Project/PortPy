from utils import *

import numpy as np
import matplotlib.pyplot as plt

def main():
    patientFolderPath = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Paraspinal_Patient_1'
    saveFilePath = r'C:\Users\fua\Documents\Figures'
    saveFileName = saveFilePath + r'\robust_smooth.jpg'
    
    # Read all the metadata for the required patient.
    metaData = loadMetaData(patientFolderPath)
    
    # Beams for different patient movements +/- (x,y,z) direction.
    # Gantry angles: 220, 240, 260, 100, 120, 140, 160, 180
    beamIndicesDict = {"Nominal": np.arange(9),
                       "Xmin3":   np.arange(9,18),
                       # "Xplus3":  np.arange(18,27),
                       # "Ymin3":   np.arange(27,36),
                       # "Yplus3":  np.arange(36,45),
                       # "Zmin3":   np.arange(45,54),
                       # "Zplus3":  np.arange(54,63)
                      }

    # Optimization parameters.
    cutoff = 0.025    
    # smoothLambda = [0, 0.15, 0.25, 0.5, 1, 10]
    smoothLambda = [0, 0.5]

    doseTrueNormMat = []
    doseDiffNormMat = []
    scenarioNameList = []
    for scenarioName, beamIndices in beamIndicesDict.items():
        # Create IMRT Plan.
        print("Solving NNLS cutoff problem for {0} scenario...".format(scenarioName))
        myPlan = createIMRTPlan(metaData, beamIndices = beamIndices)
        
        doseTrueNorm = []
        doseDiffNorm = []
        
        for lam in smoothLambda:
            print("Smoothing weight: {0}".format(lam))
            w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = runNNLSOptimization_CVX(myPlan, cutoff = cutoff, lambda_x = lam, lambda_y = lam, verbose = False)
            d_true_norm = np.linalg.norm(d_true_smooth)
            d_diff_norm = np.linalg.norm(d_cut_smooth - d_true_smooth)
            doseTrueNorm.append(d_true_norm)
            doseDiffNorm.append(d_diff_norm)
            
        doseTrueNormMat += [doseTrueNorm]
        doseDiffNormMat += [doseDiffNorm]
        scenarioNameList.append(scenarioName)
    
    # Form matrices with rows = smoothing weights, columns = movement scenarios.
    doseTrueNormMat = np.column_stack(doseTrueNormMat)
    doseDiffNormMat = np.column_stack(doseDiffNormMat)
    doseRelDiffNormMat = np.sum(doseDiffNormMat, axis = 1)/np.sum(doseTrueNormMat, axis = 1)
    
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    for j in range(len(scenarioNameList)):
        plt.plot(smoothLambda, doseRelDiffNormMat[j], label = scenarioNameList[j])
    plt.xlabel("$\lambda$")
    plt.ylabel("$\sum_{s=1}^N ||A_s^{cutoff}x_s - A_s^{true}x_s||_2/\sum_{s=1}^N ||A_s^{true}x_s||_2$")
    plt.title("Robustness Error vs. Smoothing Weight")
    plt.legend()
    plt.show()
    
    fig.savefig(saveFileName, bbox_inches = "tight", dpi = 300)

if __name__ == "__main__":
    main()
