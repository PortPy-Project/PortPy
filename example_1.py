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

    # create my_plan object for the planner beams
    # for the customized beams, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = Plan(patient_name, options=options)

    # sample methods to access data for beam with beam_id=0
    beam_PTV_mask_0 = my_plan.beams.get_structure_mask_2dgrid(beam_id=0, organ='PTV')
    beamlet_idx_2dgrid_0 = my_plan.beams.get_beamlet_idx_2dgrid(beam_id=0)

    # run IMRT optimization using cvxpy and default solver MOSEK
    # to use open source solvers e.g.ECOS using cvxpy, you can argument solver='ECOS'
    my_plan.run_optimization()

    # plot fluence
    my_plan.beams.plot_fluence_3d(beam_id=0)
    my_plan.beams.plot_fluence_2d(beam_id=0)

    ##plot dvh and robust dvh
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'BLADDER', 'BLAD_WALL', 'RECT_WALL']
    my_plan.plot_dvh(orgs=orgs)


if __name__ == "__main__":
    example_1()
