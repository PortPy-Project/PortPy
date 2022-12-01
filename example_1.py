import numpy as np

from utils.plan import Plan
import pickle


def example_1():
    # Enter patient name
    patient_name = 'ECHO_PROST_1'

    # create my_plan object for the planner beams
    # for the customized beams, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = Plan(patient_name)

    # run IMRT optimization using cvxpy and default solver MOSEK
    # to use open source solvers e.g.ECOS using cvxpy, you can argument solver='ECOS'
    my_plan.optimize.run_IMRT_optimization()

    # vox_map = my_plan._structures.down_sample_voxels(down_sample_xyz=[3, 3, 1])
    # # plot fluence
    my_plan.beams.plot_fluence_3d(beam_id=1)
    my_plan.beams.plot_fluence_2d(beam_id=1)

    # plot dvh and robust dvh
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'BLADDER', 'BLAD_WALL', 'RECT_WALL',
            'RIND_0', 'RIND_1', 'RIND_2', 'RIND_3']
    my_plan.visualize.plot_dvh(orgs=orgs)
    my_plan.visualize.plot_3d_dose()
    print('Done!')

    # # save data for debug
    # picklefile = open('my_plan', 'wb')
    # # pickle the dictionary and write it to file
    # pickle.dump(my_plan, picklefile)
    # # close the file
    # picklefile.close()

    # load pickle data
    # picklefile = open('my_plan', 'rb')
    # my_plan = pickle.load(picklefile)
    # picklefile.close()

    # sample methods to access data for beam with beam_id=1
    # beam_PTV_mask_0 = my_plan.beams.get_structure_mask_2dgrid(beam_id=1, organ='PTV')
    # beamlet_idx_2dgrid_0 = my_plan.beams.get_beamlet_idx_2dgrid(beam_id=1)

    # boolean or create margin around structures
    # st = my_plan.structures
    # my_plan.structures.union(str_1='PTV', str_2='GTV', str1_union_str2='dummy')
    # my_plan.structures.intersect(str_1='PTV', str_2='GTV', str1_union_str2='dummy')

    # PTV = st.structure('PTV')
    # GTV = st.structure('GTV')
    # st.intersect()
    # mask_3d_mod1 = PTV + GTV
    # mask_3d_mod2 = PTV * GTV  # (intersect)
    # st.modify_structure('PTV', mask_3d_mod)
    # st.add_structure('PTV-GTV', mask_3d_mod1)
    # st.add_structure('PTV_and_GTV', mask_3d_mod2)

if __name__ == "__main__":
    example_1()
