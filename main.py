
from utils import *
import os
from visualization import *
import matplotlib.pyplot as plt


def main():
    patientName = 'Lung_Patient_2'
    patientFolderPath = os.path.join(os.getcwd(), "..", 'Data', patientName)

    # read all the meta data for the required patient
    metaData = loadMetaData(patientFolderPath)

    ##create IMRT Plan

    ##options for loading requested data
    # if 1 then load the data. if 0 then skip loading the data
    options = dict()
    options['loadInfluenceMatrixFull'] = 0
    options['loadInfluenceMatrixSparse'] = 1
    options['loadBeamEyeViewStructureMask'] = 0

    beamIndices = [10, 20, 30, 40]

    myPlan = createIMRTPlan(metaData, options=options, beamIndices=beamIndices)

    w = runIMRTOptimization_CVX(myPlan)

    wMaps = getFluenceMap(myPlan, w)

    ##Plot 1st beam fluence
    (fig, ax, surf) = surface_plot(wMaps[0], cmap='viridis', edgecolor='black')
    fig.colorbar(surf)
    ax.set_zlabel('Fluence Intensity')
    plt.show()


if __name__ == "__main__":
    main()
