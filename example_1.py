from portpy.plan import Plan
from portpy.visualization import Visualization as visualize
from portpy.optimization import Optimization as optimize


def example_1():
    # Enter patient name
    patient_name = 'Lung_Patient_1'
    visualize.display_patient_metadata(patient_name)
    visualize.display_patients()

    # create my_plan object for the planner beams
    # for the customized beams, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = Plan(patient_name)
    sol = optimize.run_IMRT_fluence_map_CVXPy(my_plan)

    # # saving and loading plans
    # my_plan = Plan.load_plan()
    # sol = Plan.load_optimal_sol('sol')

    optimal_fluence_2d = my_plan.inf_matrix.fluence_1d_to_2d(sol)
    visualize.plot_fluence_3d(my_plan, beam_id=1, optimal_fluence_2d=optimal_fluence_2d)
    visualize.plot_fluence_2d(my_plan, beam_id=1, optimal_fluence_2d=optimal_fluence_2d)

    my_plan.save_plan(path=r'C:\temp')
    my_plan.save_optimal_sol(sol, sol_name='sol', path=r'C:\temp')

    # plot dvh for the structures in list
    orgs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'BLADDER', 'BLAD_WALL', 'RECT_WALL',
            'RIND_0', 'RIND_1', 'RIND_2', 'RIND_3']
    visualize.plot_dvh(my_plan, sol=sol, structs=orgs)

    # plot 2d slice for the given structures
    visualize.plot_2d_dose(my_plan, sol=sol, slice_num=50, structs=['PTV'])
    visualize.plan_metrics(my_plan, sol)

    # view ct, dose_1d and segmentations in 3d slicer.
    # First save the Nrrd images in path directory
    my_plan.save_nrrd(sol=sol, path=r'C:\temp')
    visualize.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe', img_dir=r'C:\temp')
    print('Done!')

    # sample methods to access data for beam with beam_id=1
    # beam_PTV_mask_0 = my_plan.beams.get_structure_mask_2dgrid(beam_id=1, organ='PTV')
    # beamlet_idx_2dgrid_0 = my_plan.beams.get_beamlet_idx_2dgrid(beam_id=1)

    # boolean or create margin around structures
    # my_plan.structures.union(str_1='PTV', str_2='GTV', str1_union_str2='dummy')
    # my_plan.structures.intersect(str_1='PTV', str_2='GTV', str1_union_str2='dummy')
    # my_plan.structures.expand(structure='PTV', margin_mm=5, new_structure='dummy')


if __name__ == "__main__":
    example_1()
