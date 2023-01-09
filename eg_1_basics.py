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
from portpy.plan import Plan
from portpy.visualization import Visualization as visualize
from portpy.optimization import Optimization as optimize


def eg_1_basics():

    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'C:\Users\zarepism\Google Drive\Collaborations'
    # display the existing patients
    visualize.display_patients(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams, structures, )
    patient_name = 'Lung_Patient_1'
    visualize.display_patient_metadata(patient_name, data_dir=data_dir)


    # create my_plan object for the planner beams
    # for the customized beams, you can pass the argument beam_ids
    # e.g. my_plan = Plan(patient_name, beam_ids=[0,1,2,3,4,5,6], options=options)
    my_plan = Plan(patient_name, data_folder=data_dir)

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    sol = optimize.run_IMRT_fluence_map_CVXPy(my_plan, solver='MOSEK')

    # # saving and loading plans and optimal solution
    # my_plan.save_plan(path=r'C:\temp')
    # my_plan.save_optimal_sol(sol, sol_name='sol', path=r'C:\temp')
    # my_plan = Plan.load_plan(path=r'C:\temp')
    # sol = Plan.load_optimal_sol(sol_name='sol', path=r'C:\temp')

    # plot fluence 3d and 2d
    visualize.plot_fluence_3d(sol=sol, beam_id=0)
    visualize.plot_fluence_2d(sol=sol, beam_id=0)

    # plot dvh for the structures in list
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    # plot methods are exposed using two ways:
    # 1. using visualization class
    visualize.plot_dvh(my_plan, sol=sol, structs=structs)

    # 2. Using object of class Plan
    my_plan.plot_dvh(sol=sol, structs=structs)

    # plot 2d axial slice for the given structures
    visualize.plot_2d_dose(my_plan, sol=sol, slice_num=50, structs=['PTV'])

    # visualize plan metrics based upon clinical citeria
    visualize.plan_metrics(my_plan, sol=sol)

    # view ct, dose_1d and segmentations in 3d slicer. This requires downloading and installing 3d slicer
    # First save the Nrrd images in path directory
    my_plan.save_nrrd(sol=sol, path=r'C:\temp')
    visualize.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe', img_dir=r'C:\temp')
    print('Done!')


if __name__ == "__main__":
    eg_1_basics()
