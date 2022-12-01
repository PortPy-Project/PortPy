from utils import *
import os
import visualization
import matplotlib.pyplot as plt
# from utils.plan import Plan
import pickle


def main():
    # patient_name = 'ECHO_PROST_1'
    patient_name = 'ECHO_PROST_1'
    ##create IMRT Plan

    # options for loading requested data
    # if 1 then load the data. if 0 then skip loading the data
    options = dict()
    options['loadInfluenceMatrixFull'] = 0
    options['loadInfluenceMatrixSparse'] = 1
    options['loadBeamEyeViewStructureMask'] = 1

    # beam_indices = [46, 131, 36, 121, 26, 66, 151, 56, 141]
    my_plan = Plan(patient_name, options=options)
    # get functions
    a = my_plan.beams.get_structure_mask_2dgrid(beam_id=0, organ='PTV')
    b = my_plan.beams.get_beamlet_idx_2dgrid(beam_id=0)

    # run optimization
    my_plan.run_IMRT_optimization()

    # save data for debug mode
    picklefile = open('my_plan', 'wb')
    # pickle the dictionary and write it to file
    pickle.dump(my_plan, picklefile)
    # close the file
    picklefile.close()

    # load pickle data
    picklefile = open('my_plan', 'rb')
    my_plan = pickle.load(picklefile)
    picklefile.close()

    my_plan.beams.plot_fluence_3d(beam_id=0)
    my_plan.beams.plot_fluence_2d(beam_id=0)

    ##plot dvh and robust dvh
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'BLADDER', 'BLAD_WALL', 'RECT_WALL']
    my_plan.plot_dvh(orgs=orgs)

    # dose_list = [my_plan['infMatrixSparse'] * w, my_plan['infMatrixSparse'] * w * 1.05,
    #              my_plan['infMatrixSparse'] * w * 0.95]
    # visualization.plot_robust_dvh(dose_list, my_plan, orgs=orgs, plot_scenario=[0])
    # visualization.plot_dvh(my_plan['infMatrixSparse']*w, my_plan, orgs=orgs)

    ##Plot 1st beam fluence
    # wMaps = visualization.get_fluence_map(my_plan, w)
    # (fig, ax, surf) = visualization.surface_plot(wMaps[0], cmap='viridis', edgecolor='black')
    # fig.colorbar(surf)
    # ax.set_zlabel('Fluence Intensity')
    # plt.show()
    # my_plan.plot_dvh()


if __name__ == "__main__":
    main()
