
from utils import *
import os
import visualization
import matplotlib.pyplot as plt

def main():
    patient_name = 'Lung_Patient_1'
    patient_folder_path = os.path.join(os.getcwd(), "..", 'Data', patient_name)

    # read all the meta data for the required patient
    meta_data = load_metadata(patient_folder_path)

    ##create IMRT Plan

    ##options for loading requested data
    # if 1 then load the data. if 0 then skip loading the data
    options = dict()
    options['loadInfluenceMatrixFull'] = 0
    options['loadInfluenceMatrixSparse'] = 1
    options['loadBeamEyeViewStructureMask'] = 0

    # beam_indices = [46, 131, 36, 121, 26, 66, 151, 56, 141]
    beam_indices = [10, 20, 30, 50]
    my_plan = create_imrt_plan(meta_data, options=options, beam_indices=beam_indices)

    w = run_imrt_optimization_cvx(my_plan)

    ##plot dvh and robust dvh
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD']
    dose_list = [my_plan['infMatrixSparse'] * w, my_plan['infMatrixSparse'] * w * 1.05,
                 my_plan['infMatrixSparse'] * w * 0.95]
    visualization.plot_robust_dvh(dose_list, my_plan, orgs=orgs, norm_flag=True)
    visualization.plot_dvh(my_plan['infMatrixSparse']*w, my_plan, orgs=orgs)

    ##Plot 1st beam fluence
    wMaps = visualization.get_fluence_map(my_plan, w)
    (fig, ax, surf) = visualization.surface_plot(wMaps[0], cmap='viridis', edgecolor='black')
    fig.colorbar(surf)
    ax.set_zlabel('Fluence Intensity')
    plt.show()


if __name__ == "__main__":
    main()
