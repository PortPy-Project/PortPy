"""

This example demonstrates performing the following tasks using portpy:
    1- Query the existing patients in the database
        (you first need to download the patient database from the link provided in the GitHub page).
    2- Query the data provided for a specified patient in the database.
    3- Create a simple IMRT plan using CVXPy package. You can call different opensource/commercial optimization engines
        from CVXPy,but you first need to download them and obtain an appropriate license.
        Most commercial optimization engines (e.g., Mosek, Gorubi) give free academic license if you have .edu email
        address
    4- Visualise the plan (dose distribution, fluence)
    5- Evaluate the plan based on some clinically relevant metrics


"""
import portpy as pp


def ex_1_basics():
    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'F:\Research\Data_newformat\Python-PORT\Data'
    # display the existing patients. To display it in browser rather than console, turn on in_browser=True
    pp.Visualize.display_patients(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, )
    patient_name = 'Lung_Patient_1'
    pp.Visualize.display_patient_metadata(patient_name, data_dir=data_dir)

    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = pp.Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = pp.Plan(patient_name)

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    sol = pp.Optimize.run_IMRT_fluence_map_CVXPy(my_plan, solver='MOSEK')

    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan', path=r'C:\temp')
    pp.save_optimal_sol(sol, sol_name='sol', path=r'C:\temp')
    my_plan = pp.load_plan(path=r'C:\temp')
    sol = pp.load_optimal_sol(sol_name='sol', path=r'C:\temp')

    # plot fluence 3d and 2d
    pp.Visualize.plot_fluence_3d(sol=sol, beam_id=0)
    pp.Visualize.plot_fluence_2d(sol=sol, beam_id=0)

    # plot dvh for the structures in the given list. Default dose is in Gy and volume is in relative scale(%).
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    # visualization methods are exposed using two ways:
    # 1. using visualization class
    pp.Visualize.plot_dvh(my_plan, sol=sol, structs=structs)

    # 2. Using object of class Plan. In this case, one doesnt need to pass the plan object as an argument
    my_plan.plot_dvh(sol=sol, structs=structs)

    # plot 2d axial slice for the given solution and display the structures on the slice
    pp.Visualize.plot_2d_dose(my_plan, sol=sol, slice_num=40, structs=['PTV'])

    # visualize plan metrics based upon clinical criteria
    pp.Visualize.plan_metrics(my_plan, sol=sol)

    # view ct, dose_1d and segmentations in 3d slicer. This requires downloading and installing 3d slicer
    # First save the Nrrd images in data_dir directory
    pp.save_nrrd(my_plan, sol=sol, data_dir=r'C:\temp')
    pp.Visualize.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe',
                                data_dir=r'C:\temp')
    print('Done!')


if __name__ == "__main__":
    ex_1_basics()
