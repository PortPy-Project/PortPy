"""
    This example demonstrates the use of portpy to create down sampled IMRT plan and
    optimize it with exact dvh constraints for benchmarking
"""
import portpy as pp


def eg_4_dvh_benchmark_optimization():
    # Enter patient name
    patient_name = 'Lung_Patient_1'

    # visualize patient metadata for beams_dict and structures
    pp.Visualize.display_patient_metadata(patient_name)

    # display patients
    pp.Visualize.display_patients()

    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = pp.Plan(patient_name)

    # create a influence matrix down sampled beamlets of width and height 5mm
    inf_matrix_55_772 = my_plan.create_inf_matrix(beamlet_width_mm=5, beamlet_height_mm=5, down_sample_xyz=[7, 7, 2])

    # run IMRT optimization using cvxpy and default solver MOSEK
    # to use open source solvers e.g.ECOS using cvxpy, you can argument solver='ECOS'
    # get optimal solution for default
    sol_1 = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan)

    # optimize with downscaled influence matrix and exact dvh constraint
    sol_55_772 = pp.Optimize.run_IMRT_fluence_map_CVXPy_dvh_benchmark(my_plan, inf_matrix=inf_matrix_55_772)

    # Comment/Uncomment these lines to save & load plan and optimal solutions
    my_plan.save_plan(path=r'C:\temp')
    my_plan.save_optimal_sol(sol_55_772, sol_name='sol_55_772', path=r'C:\temp')
    my_plan.save_optimal_sol(sol_1, sol_name='sol_1', path=r'C:\temp')
    # my_plan = Plan.load_plan(path=r'C:\temp')
    # sol_1 = Plan.load_optimal_sol('sol_1', path=r'C:\temp')
    # sol_2 = Plan.load_optimal_sol('sol_55_772', path=r'C:\temp')

    # plot fluence in 3d
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_1)
    pp.Visualize.plot_fluence_3d(beam_id=0, sol=sol_55_772)

    # plot fluence in 2d
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_1)
    pp.Visualize.plot_fluence_2d(beam_id=0, sol=sol_55_772)

    #

    # plot dvh dvh for both the cases
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    pp.Visualize.plot_dvh(my_plan, sol=sol_1, structs=structs, style='solid', show=False)
    pp.Visualize.plot_dvh(my_plan, sol=sol_55_772, structs=structs, style='dotted', create_fig=False)

    # visualize 2d dose for both the cases
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_1)
    pp.Visualize.plot_2d_dose(my_plan, sol=sol_55_772)

    print('Done!')


if __name__ == "__main__":
    eg_4_dvh_benchmark_optimization()
