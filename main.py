
from utils import *
import os
from visualization import *
import matplotlib.pyplot as plt


def main():
    patientName = 'Lung_Patient_1'
    patientFolderPath = os.path.join(os.getcwd(), "..", 'Data', patientName)

    # read all the meta data for the required patient
    metaData = loadMetaData(patientFolderPath)

    ##create IMRT Plan
    # gantryRtns = [8, 16, 24, 32, 80, 120]
    # collRtns = [0, 0, 0, 90, 0, 90]
    # beamIndices = [46,131,36,121,26,66,151,56,141]
    beamIndices = [10, 20, 30, 40]

    myPlan = createIMRTPlan(metaData, beamIndices=beamIndices)

    w = runIMRTOptimization_CVX(myPlan)

    wMaps = getFluenceMap(myPlan, w)

    ##Plot 1st beam fluence
    (fig, ax, surf) = surface_plot(wMaps[1], cmap='viridis', edgecolor='black')
    fig.colorbar(surf)
    ax.set_zlabel('Fluence Intensity')
    plt.show()


if __name__ == "__main__":
    main()
