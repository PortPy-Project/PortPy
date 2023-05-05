"""

This example demonstrates the following main functionalities of portpy_photon:
    1- Accessing the curated portpy data
        (DataExplorer class)

    2- Creating a simple IMRT plan
        (Plan class, Optimization class)

    3- Visualising the plan (e.g., dose distribution, fluence, DVH)
       (Visualization class)

    4- Evaluating the plan (e.g., max/mean/DVH points, established clinical metrics)
        (Evaluation class)

    5- Saving the plan and solution for future uses
        (Utils)

    6- Advanced plan visualization through integration with a popular 3D-Slicer package
        (Visualization class)


"""
import numpy as np
import portpy.photon as pp


def ex_1_introduction():
    # ***************** 1) accessing the portpy data (DataExplorer class)**************************
    # Note: you first need to download the patient database from the link provided in the GitHub page.

    # specify the patient data location.
    data_dir = r'../../data'
    # display the existing patients in console or browser.
    pp.Visualize.display_patients(data_dir=data_dir)
    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    patient_id = 'Lung_Phantom_Patient_1'
    # display the data of the patient in console or browser.
    pp.Visualize.display_patient_metadata(patient_id, data_dir=data_dir)

    # ***************** 2) creating a simple IMRT plan using CVXPy (Plan class, Optimization class)*********************
    # Note: you can call different opensource / commercial optimization engines from CVXPy.
    #   For commercial engines (e.g., Mosek, Gorubi, CPLEX), you first need to obtain an appropriate license.
    #   Most commercial optimization engines give free academic license.

    # Create my_plan object which would load and store all the data needed for optimization
    #   (e.g., influence matrix, structures and their voxels, beams and their beamlets).
    # If the list of beams are not provided, it uses the beams selected manually
    #   by a human expert planner for the patient (manually selected beams are stored in portpy data).
    my_plan = pp.Plan(patient_id, data_dir=data_dir)
    # Creating rinds (aka rings, shells)
    # Rinds are doughnut-shaped structures often created to control the radiation dose to non-specified structures
    #   They can also control the dose fall-off after PTV.
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

    # ***************** 3) visualizing the plan (Visualization class) *****************
    # plot fluence 3d and 2d for the 1st beam
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
    pp.Visualize.plot_2d_dose(my_plan, sol=sol, slice_num=60, structs=['PTV'])

    # ***************** 4) evaluating the plan (Evaluation class) *****************
    # 4) visualizing the results
    # visualize plan metrics based upon clinical criteria
    pp.Visualize.plan_metrics(my_plan, sol=sol)

    # ***************** 5) Advanced visualization with 3D-Slicer (Visualization class) *****************
    # Note: This requires downloading and installing 3d slicer
    #   3D slicer integration is seamless with Jupiter Notebook (see ex_7_Slicer)
    # Without Jupiter Notebook, we first need to save our data (ct, 3D dose, structures)
    #   as images in nrrd format on disk and then lunch Slicer by providing data as input argument
    pp.save_nrrd(my_plan, sol=sol, data_dir=r'C:\temp')
    pp.Visualize.view_in_slicer(my_plan, slicer_path=r'C:\ProgramData\NA-MIC\Slicer 5.2.2\Slicer.exe',
                                data_dir=r'C:\temp')
    print('Done!')


if __name__ == "__main__":
    ex_1_introduction()
