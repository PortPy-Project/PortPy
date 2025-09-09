# tests/test_basic_tutorial.py
import os
import matplotlib
matplotlib.use("Agg")   # Use non-GUI backend

import numpy as np
import pytest
import portpy.photon as pp

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "portpy_test_data"))


@pytest.fixture(scope="module")
def data_explorer():
    """
    Fixture to initialize DataExplorer with real data.
    Downloads Lung_Phantom_Patient_1 if not present.
    """
    data = pp.DataExplorer(hf_repo_id="PortPy-Project/PortPy_Dataset", local_download_dir=DATA_DIR)

    # Pick the patient for testing
    data.patient_id = "Lung_Phantom_Patient_1"
    print("Downloading dataset if needed...")
    data.filter_and_download_hf_dataset()  # ensure data is downloaded
    return data


@pytest.fixture(scope="module")
def ct_structs_beams(data_explorer):
    """
    Fixture to initialize CT, Structures, and Beams objects.
    """
    ct = pp.CT(data_explorer)
    structs = pp.Structures(data_explorer)
    beams = pp.Beams(data_explorer)
    return ct, structs, beams


@pytest.fixture(scope="module")
def plan_and_solution(data_explorer, ct_structs_beams):
    ct, structs, beams = ct_structs_beams

    # Load clinical criteria
    protocol_name = "Lung_2Gy_30Fx"
    clinical_criteria = pp.ClinicalCriteria(data_explorer, protocol_name=protocol_name)

    # Optimization parameters and structures
    opt_params = data_explorer.load_config_opt_params(protocol_name=protocol_name)
    structs.create_opt_structures(opt_params=opt_params, clinical_criteria=clinical_criteria)

    # Influence matrix
    inf_matrix = pp.InfluenceMatrix(ct=ct, structs=structs, beams=beams)

    # Create plan and optimization
    my_plan = pp.Plan(ct=ct, structs=structs, beams=beams, inf_matrix=inf_matrix,
                      clinical_criteria=clinical_criteria)
    opt = pp.Optimization(my_plan, opt_params=opt_params)
    opt.create_cvxpy_problem()
    sol = opt.solve(solver="SCS", verbose=False)  # use free solver to avoid license issues

    return my_plan, sol


def test_patient_loaded(data_explorer):
    assert data_explorer.patient_id == "Lung_Phantom_Patient_1"
    # assert "Lung_Phantom_Patient_1" in data_explorer.list_of_patients()  # or display_list_of_patients()


def test_plan_creation(plan_and_solution):
    plan, sol = plan_and_solution
    assert hasattr(plan, "beams")
    assert len(plan.beams.get_all_beam_ids()) > 0
    assert sol is not None
    assert np.isclose(sol['obj_value'], 54.50, rtol=0.0001)


def test_visualization_methods(plan_and_solution):
    plan, sol = plan_and_solution
    beam_id = plan.beams.get_all_beam_ids()[0]

    # Plotting should run without error (we don't need to check plots)
    pp.Visualization.plot_fluence_2d(sol=sol, beam_id=beam_id)
    pp.Visualization.plot_fluence_3d(sol=sol, beam_id=beam_id)
    pp.Visualization.plot_dvh(plan, sol=sol, struct_names=['PTV'])
    pp.Visualization.plot_2d_slice(my_plan=plan, sol=sol, slice_num=60, struct_names=['PTV'])


def test_evaluation(plan_and_solution):
    plan, sol = plan_and_solution
    # Evaluation should run without errors
    pp.Evaluation.display_clinical_criteria(plan, sol=sol)
