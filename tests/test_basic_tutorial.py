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
    clinical_criteria.get_dvh_table(my_plan=my_plan, opt_params=opt_params)
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


def test_cvar_constraints_satisfaction(plan_and_solution):
    """
    Verify that the optimized dose satisfies the CVaR constraints
    defined in the clinical criteria.
    """
    plan, sol = plan_and_solution
    inf_matrix = plan.inf_matrix
    clinical_criteria = plan.clinical_criteria
    dvh_table = clinical_criteria.dvh_table

    # Get the per-fraction beamlet weights (optimal fluence)
    # PortPy 'optimal_intensity' is per-fraction
    x = sol['optimal_intensity']

    # Iterate through the table to find constraints that were set as CVaR
    for ind in dvh_table.index:
        if dvh_table.get('dvh_method', {}).get(ind) == 'cvar':
            org = dvh_table['structure_name'][ind]
            bound_type = dvh_table.get('bound_type', {}).get(ind, 'upper')

            # 1. Get per-fraction dose for this structure
            voxels_idx = inf_matrix.get_opt_voxels_idx(org)
            voxels_cc = inf_matrix.get_opt_voxels_volume_cc(org)
            # Dose per fraction = A_matrix[voxels] * x
            struct_dose = inf_matrix.A[voxels_idx, :] @ x

            # 2. Get the limit and tail fraction
            # Remember to adjust for volume in calc box as done in opt
            fov_factor = inf_matrix.get_fraction_of_vol_in_calc_box(org)
            limit_gy_per_frac = dvh_table['dose_gy'][ind]/plan.get_num_of_fractions()
            volume_perc = dvh_table['volume_perc'][ind] / fov_factor

            if bound_type == 'upper':
                tail_fraction = volume_perc / 100.0
            else:
                tail_fraction = 1.0 - (volume_perc / 100.0)

            # 3. Calculate CVaR manually from the dose distribution
            # Sort dose to find the tail
            sorted_indices = np.argsort(struct_dose)
            if bound_type == 'upper':
                # Hottest tail: highest doses
                sorted_dose = struct_dose[sorted_indices[::-1]]
                sorted_cc = voxels_cc[sorted_indices[::-1]]
            else:
                # Coldest tail: lowest doses
                sorted_dose = struct_dose[sorted_indices]
                sorted_cc = voxels_cc[sorted_indices]

            # Calculate cumulative volume to find the cutoff
            cum_vol = np.cumsum(sorted_cc)
            total_vol = np.sum(voxels_cc)
            tail_vol_limit = tail_fraction * total_vol

            # Find voxels belonging to the tail
            tail_mask = cum_vol <= tail_vol_limit

            # If tail_vol_limit is very small (e.g. 1 voxel), ensure we take at least one
            if not np.any(tail_mask):
                actual_cvar = sorted_dose[0]
            else:
                # Simple volume-weighted average of the tail
                actual_cvar = np.sum(sorted_dose[tail_mask] * sorted_cc[tail_mask]) / np.sum(sorted_cc[tail_mask])

            # 4. Assert with a small tolerance for solver noise
            # SCS is a first-order solver, so use a reasonable tolerance (e.g., 1%)
            if bound_type == 'upper':
                # CVaR (hottest) should be <= limit
                assert actual_cvar <= limit_gy_per_frac * 1.01, \
                    f"Upper CVaR failed for {org}. Limit: {limit_gy_per_frac}, Actual: {actual_cvar}"
            else:
                # CVaR (coldest) should be >= limit
                assert actual_cvar >= limit_gy_per_frac * 0.99, \
                    f"Lower CVaR failed for {org}. Limit: {limit_gy_per_frac}, Actual: {actual_cvar}"

            print(f"Verified {bound_type} CVaR for {org}: {actual_cvar:.4f} vs {limit_gy_per_frac:.4f}")
