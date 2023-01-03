from portpy.plan import Plan
from portpy.visualization import Visualization as visualize
from portpy.optimization import Optimization as optimize


def eg_1_basics():
    # Enter patient name
    patient_name = 'Lung_Patient_1'

    # visualize patient metadata for beams and structures
    visualize.display_patient_metadata(patient_name)

    # display patients
    visualize.display_patients()

    # create my_plan object for the planner beams
    # for the customized beams, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = Plan(patient_name)

    # run imrt fluence map optimization using cvxpy and save the optimal solution in sol
    sol = optimize.run_IMRT_fluence_map_CVXPy(my_plan)

    # # saving and loading plans and optimal solution
    # my_plan.save_plan(path=r'C:\temp')
    # my_plan.save_optimal_sol(sol, sol_name='sol', path=r'C:\temp')
    # my_plan = Plan.load_plan(path=r'C:\temp')
    # sol = Plan.load_optimal_sol(sol_name='sol', path=r'C:\temp')

    visualize.plot_fluence_3d(my_plan, sol=sol, beam_id=0)
    visualize.plot_fluence_2d(my_plan, sol=sol, beam_id=0)

    # plot dvh for the structures in list
    structs = ['PTV', 'CTV', 'GTV', 'ESOPHAGUS', 'HEART', 'CORD', 'BLADDER', 'BLAD_WALL', 'RECT_WALL',
               'RIND_0', 'RIND_1', 'RIND_2', 'RIND_3']
    # plot methods is exposed using two ways:
    # 1. using visualization class
    visualize.plot_dvh(my_plan, sol=sol, structs=structs)

    # 2. Using object of class Plan
    my_plan.plot_dvh(sol=sol, structs=structs)

    # plot 2d axial slice for the given structures
    visualize.plot_2d_dose(my_plan, sol=sol, slice_num=50)

    # visualize plan metrics based upon clinical citeria
    visualize.plan_metrics(my_plan, sol)

    # view ct, dose_1d and segmentations in 3d slicer.
    # First save the Nrrd images in path directory
    my_plan.save_nrrd(sol=sol, path=r'C:\temp')
    visualize.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe', img_dir=r'C:\temp')
    print('Done!')


if __name__ == "__main__":
    eg_1_basics()
