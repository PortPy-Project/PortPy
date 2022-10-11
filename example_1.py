import numpy as np

from utils import *


def example_1():
    # Enter patient name
    patient_name = 'Lung_Patient_1'

    # options for loading requested data
    # if 1 then load the data. if 0 then skip loading the data
    options = dict()
    options['loadInfluenceMatrixFull'] = 0
    options['loadInfluenceMatrixSparse'] = 1
    options['loadBeamEyeViewStructureMask'] = 1

    # Enter the beam indices for creating plan
    beam_ids = np.arange(0, 7)

    # create plan object
    my_plan = Plan(patient_name, beam_ids=beam_ids, options=options)

    # sample methods to access data
    beam_PTV_mask_0 = my_plan.beams.get_structure_mask_2dgrid(beam_id=0, organ='PTV')
    beamlet_idx_2dgrid_0 = my_plan.beams.get_beamlet_idx_2dgrid(beam_id=0)

    # run optimization
    my_plan.run_optimization()

    # plot fluence
    my_plan.beams.plot_fluence_3d(beam_id=0)
    my_plan.beams.plot_fluence_2d(beam_id=0)

    ##plot dvh and robust dvh
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'BLADDER', 'BLAD_WALL', 'RECT_WALL']
    my_plan.plot_dvh(orgs=orgs)


if __name__ == "__main__":
    example_1()
