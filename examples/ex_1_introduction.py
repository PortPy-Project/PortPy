"""

This example demonstrates performing the following tasks using portpy_photon:
    1- Query the existing patients in the database
        (you first need to download the patient database from the link provided in the GitHub page).
    2- Query the data provided for a specified patient in the database.
    3- Create a simple IMRT plan using CVXPy package. You can call different opensource/commercial optimization engines
        from CVXPy,but you first need to download them and obtain an appropriate license.
        Most commercial optimization engines (e.g., Mosek, Gorubi) give free academic license if you have .edu email
        address
    4- Visualise the plan (dose_1d distribution, fluence)
    5- Evaluate the plan based on some clinically relevant metrics


"""
import portpy_photon as pp
import numpy as np


def ex_1_introduction():
    # specify the patient data location
    # (you first need to download the patient database from the link provided in the GitHub page)
    data_dir = r'../data'
    # display the existing patients. To display it in browser rather than console, turn on in_browser=True
    # pp.Visualize.display_patients(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info about the patient data (e.g., beams_dict, structures, )
    patient_id = 'Lung_Phantom_Patient_2'
    pp.Visualize.display_patient_metadata(patient_id, data_dir=data_dir)

    # create my_plan object for the planner beams_dict
    # for the customized beams_dict, you can pass the argument beam_ids
    # e.g. my_plan = pp.Plan(patient_name, beam_ids=[0,10,20,30,40,50,60])
    my_plan = pp.Plan(patient_id, data_dir=data_dir)

    # Let us create rinds for creating reasonable dose fall off for the plan
    rind_max_dose = np.array([1.1, 1.05, 0.9, 0.85, 0.75]) * my_plan.get_prescription()
    rind_params = [{'rind_name': 'RIND_0', 'ref_structure': 'PTV', 'margin_start_mm': 0, 'margin_end_mm': 5, 'max_dose_gy': rind_max_dose[0]},
                   {'rind_name': 'RIND_1', 'ref_structure': 'PTV', 'margin_start_mm': 5, 'margin_end_mm': 10, 'max_dose_gy': rind_max_dose[1]},
                   {'rind_name': 'RIND_2', 'ref_structure': 'PTV', 'margin_start_mm': 10, 'margin_end_mm': 30, 'max_dose_gy': rind_max_dose[2]},
                   {'rind_name': 'RIND_3', 'ref_structure': 'PTV', 'margin_start_mm': 30, 'margin_end_mm': 60, 'max_dose_gy': rind_max_dose[3]},
                   {'rind_name': 'RIND_4', 'ref_structure': 'PTV', 'margin_start_mm': 60, 'margin_end_mm': 'inf', 'max_dose_gy': rind_max_dose[4]}]
    my_plan.add_rinds(rind_params=rind_params)

    # create cvxpy problem using the clinical criteria
    prob = pp.CvxPyProb(my_plan)

    # run imrt fluence map optimization using cvxpy and one of the supported solvers and save the optimal solution in sol
    # CVXPy supports several opensource (ECOS, OSQP, SCS) and commercial solvers (e.g., MOSEK, GUROBI, CPLEX)
    # For optimization problems with non-linear objective and/or constraints, MOSEK often performs well
    # For mixed integer programs, GUROBI/CPLEX are good choices
    # If you have .edu email address, you can get free academic license for commercial solvers
    # we recommend the commercial solver MOSEK as your solver for the problems in this example,
    # however, if you don't have a license, you can try opensource/free solver SCS or ECOS
    # see https://www.cvxpy.org/tutorial/advanced/index.html for more info about CVXPy solvers
    # To set up mosek solver, you can get mosek license file using edu account and place the license file in directory C:\Users\username\mosek
    prob.solve(solver='MOSEK', verbose=True)
    sol = prob.get_sol()

    # Comment/Uncomment these lines to save and load the pickle file for plans and optimal solution from the directory
    pp.save_plan(my_plan, plan_name='my_plan_phantom', path=r'C:\temp')
    pp.save_optimal_sol(sol, sol_name='sol_phantom', path=r'C:\temp')
    # my_plan = pp.load_plan(path=r'C:\temp')
    # sol = pp.load_optimal_sol(sol_name='sol', path=r'C:\temp')

    # plot fluence 3d and 2d for the 1st and 2nd beam
    pp.Visualize.plot_fluence_3d(sol=sol, beam_id=my_plan.beams.get_all_beam_ids()[0])

    pp.Visualize.plot_fluence_2d(sol=sol, beam_id=my_plan.beams.get_all_beam_ids()[0])

    # plot dvh for the structures in the given list. Default dose_1d is in Gy and volume is in relative scale(%).
    structs = ['PTV', 'ESOPHAGUS', 'HEART', 'CORD']

    # visualization methods are exposed using two ways:
    # 1. using visualization class
    pp.Visualize.plot_dvh(my_plan, sol=sol, structs=structs, title=patient_id)

    # 2. Using object of class Plan. In this case, one doesnt need to pass the plan object as an argument
    my_plan.plot_dvh(sol=sol, structs=structs)

    # plot 2d axial slice for the given solution and display the structures contours on the slice
    pp.Visualize.plot_2d_dose(my_plan, sol=sol, slice_num=70, structs=['PTV'])

    # visualize plan metrics based upon clinical criteria
    pp.Visualize.plan_metrics(my_plan, sol=sol)

    # view ct, dose_1d and segmentations in 3d slicer. This requires downloading and installing 3d slicer
    # First save the Nrrd images in data_dir directory
    pp.save_nrrd(my_plan, sol=sol, data_dir=r'C:\temp')
    pp.Visualize.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe',
                                data_dir=r'C:\temp')
    print('Done!')


if __name__ == "__main__":
    ex_1_introduction()
