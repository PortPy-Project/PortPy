
from utils import *
import os
import visualization
import matplotlib.pyplot as plt
# from utils.plan import Plan

def main():
    patient_name = 'ECHO_PROST_1'

    ##create IMRT Plan

    ##options for loading requested data
    # if 1 then load the data. if 0 then skip loading the data
    options = dict()
    options['loadInfluenceMatrixFull'] = 0
    options['loadInfluenceMatrixSparse'] = 1
    options['loadBeamEyeViewStructureMask'] = 1

    # beam_indices = [46, 131, 36, 121, 26, 66, 151, 56, 141]
    my_plan = Plan(patient_name, options=options)

    w = run_imrt_optimization_cvx(my_plan)

    ##plot dvh and robust dvh
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD']
    dose_list = [my_plan['infMatrixSparse'] * w, my_plan['infMatrixSparse'] * w * 1.05,
                 my_plan['infMatrixSparse'] * w * 0.95]
    visualization.plot_robust_dvh(dose_list, my_plan, orgs=orgs, plot_scenario=[0])
    visualization.plot_dvh(my_plan['infMatrixSparse']*w, my_plan, orgs=orgs)

    ##Plot 1st beam fluence
    wMaps = visualization.get_fluence_map(my_plan, w)
    (fig, ax, surf) = visualization.surface_plot(wMaps[0], cmap='viridis', edgecolor='black')
    fig.colorbar(surf)
    ax.set_zlabel('Fluence Intensity')
    plt.show()


if __name__ == "__main__":
    main()
