"""
    This example demonstrates the use of portpy to create down sampled IMRT plan and compare it with plan
    with original resolution using different visualizations
"""
from portpy.plan import Plan
from portpy.visualization import Visualization as visualize
from portpy.optimization import Optimization as optimize


def eg_2_down_sampling():
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

    # create a influence matrix down sampled beamlets of width and height 5mm
    inf_matrix_2 = my_plan.create_inf_matrix(beamlet_width=5, beamlet_height=5)

    # create another influence matrix for down sampled voxels
    inf_matrix_3 = my_plan.create_inf_matrix(down_sample_xyz=[5, 5, 1])

    # run IMRT optimization using cvxpy and default solver MOSEK
    # to use open source solvers e.g.ECOS using cvxpy, you can argument solver='ECOS'
    # get optimal solution for default and downsampled influence matrix
    sol_1 = optimize.run_IMRT_fluence_map_CVXPy(my_plan)
    sol_2 = optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_2)
    sol_3 = optimize.run_IMRT_fluence_map_CVXPy(my_plan, inf_matrix=inf_matrix_3)

    # # Uncomment these lines to save & load plan and optimal solutions
    # my_plan.save_plan(path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_3, sol_name='sol_3', path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_2, sol_name='sol_2', path=r'C:\temp')
    # my_plan.save_optimal_sol(sol_1, sol_name='sol_1', path=r'C:\temp')
    # my_plan = Plan.load_plan(path=r'C:\temp')
    # sol_1 = Plan.load_optimal_sol('sol_1', path=r'C:\temp')
    # sol_2 = Plan.load_optimal_sol('sol_2', path=r'C:\temp')
    # sol_3 = Plan.load_optimal_sol('sol_3', path=r'C:\temp')

    # # plot fluence
    visualize.plot_fluence_3d(beam_id=0, sol=sol_1)
    visualize.plot_fluence_3d(beam_id=0, sol=sol_2)
    visualize.plot_fluence_3d(beam_id=0, sol=sol_3)
    visualize.plot_fluence_2d(beam_id=0, sol=sol_1)
    visualize.plot_fluence_2d(beam_id=0, sol=sol_2)
    visualize.plot_fluence_2d(beam_id=0, sol=sol_3)
    #

    # plot dvh dvh for both the cases
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    visualize.plot_dvh(my_plan, sol=sol_1, structs=structs, style='solid', show=False)
    visualize.plot_dvh(my_plan, sol=sol_2, structs=structs, style='dotted', create_fig=False)
    visualize.plot_dvh(my_plan, sol=sol_3, structs=structs, style='dashed', create_fig=False)

    # visualize 2d dose for both the cases
    visualize.plot_2d_dose(my_plan, sol=sol_1)
    visualize.plot_2d_dose(my_plan, sol=sol_2)
    visualize.plot_2d_dose(my_plan, sol=sol_3)

    print('Done!')


if __name__ == "__main__":
    eg_2_down_sampling()
