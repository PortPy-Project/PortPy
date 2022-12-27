from portpy.plan import Plan
from portpy.visualization import Visualization as visualize
from portpy.optimization import Optimization as optimize


def example_2():
    # Enter patient name
    patient_name = 'Lung_Patient_1'
    visualize.display_patient_metadata(patient_name)
    visualize.display_patients()
    # create my_plan object for the planner beams
    # for the customized beams, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = Plan(patient_name)

    # create a influence matrix down sampled beamlets of width and height 5mm
    inf_matrix_2 = my_plan.create_inf_matrix(beamlet_width=5, beamlet_height=5)

    # run IMRT optimization using cvxpy and default solver MOSEK
    # to use open source solvers e.g.ECOS using cvxpy, you can argument solver='ECOS'
    # get optimal solution for default and downsampled influence matrix
    sol_1 = optimize.run_IMRT_fluence_map_CVXPy(my_plan)
    sol_2 = optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_2)

    # # plot fluence
    optimal_fluence_2d_down_sample = inf_matrix_2.fluence_1d_to_2d(sol_2)
    optimal_fluence_2d = my_plan.inf_matrix.fluence_1d_to_2d(sol_1)
    visualize.plot_fluence_3d(my_plan, beam_id=1, optimal_fluence_2d=optimal_fluence_2d)
    visualize.plot_fluence_2d(my_plan, beam_id=1, optimal_fluence_2d=optimal_fluence_2d)
    visualize.plot_fluence_3d(my_plan, beam_id=1, optimal_fluence_2d=optimal_fluence_2d_down_sample)
    visualize.plot_fluence_2d(my_plan, beam_id=1, optimal_fluence_2d=optimal_fluence_2d_down_sample)

    # save plan and optimal solution
    my_plan.save_plan()
    my_plan.save_optimal_sol(sol_2, sol_name='sol_2')
    my_plan.save_optimal_sol(sol_1, sol_name='sol_1')
    # my_plan = Plan.load_plan()
    # plot dvh and robust dvh
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'BLADDER', 'BLAD_WALL', 'RECT_WALL',
            'RIND_0', 'RIND_1', 'RIND_2', 'RIND_3']
    visualize.plot_dvh(my_plan, sol=sol_1, structs=orgs, options_fig={'style': 'solid', 'show': False})
    visualize.plot_dvh(my_plan, sol=sol_1, options_fig={'style': 'dotted', 'show': True})

    visualize.plot_2d_dose(my_plan, sol=sol_1)
    visualize.plot_2d_dose(my_plan, sol=sol_2)

    # view ct, dose_1d and segmentations in 3d slicer.
    # First save the Nrrd images in path directory
    my_plan.save_nrrd(sol=sol_1, path=r'C:\temp')
    visualize.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe',
                             img_dir=r'C:\temp')
    print('Done!')

    # sample methods to access data for beam with beam_id=1
    # beam_PTV_mask_0 = my_plan.beams.get_structure_mask_2dgrid(beam_id=1, organ='PTV')
    # beamlet_idx_2dgrid_0 = my_plan.beams.get_beamlet_idx_2dgrid(beam_id=1)

    # boolean or create margin around structures
    # my_plan.structures.union(str_1='PTV', str_2='GTV', str1_union_str2='dummy')
    # my_plan.structures.intersect(str_1='PTV', str_2='GTV', str1_union_str2='dummy')
    # my_plan.structures.expand(structure='PTV', margin_mm=5, new_structure='dummy')


if __name__ == "__main__":
    example_2()
