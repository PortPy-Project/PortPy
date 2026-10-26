# PortPy examples

Every example is a Jupyter notebook; plain-Python versions of most of them are in
[`python_files/`](./python_files). The **Needs** column lists requirements beyond
`pip install portpy`. All notebooks use MOSEK by default; any CVXPY solver works for the
convex problems (the free solvers SCS and Clarabel are installed with PortPy, but are considerably slower
than MOSEK or Gurobi, which offer free academic licenses).

Most notebooks download their patient from Hugging Face on first run (about 2 GB per patient with
the expert-selected beams). See the [Data](../README.md#data) section of the main README.

## Start here

| Notebook | What you learn | Needs |
|---|---|---|
| [1_basic_tutorial](./1_basic_tutorial.ipynb) | Load a patient, create an IMRT plan, visualize and evaluate it, save and reload | – |
| [structure_operations](./structure_operations.ipynb) | Union, intersection, subtraction, expansion and shrinking of structures | – |
| [inf_matrix_down_sampling](./inf_matrix_down_sampling.ipynb) | Down-sample beamlets and voxels for faster experiments, and what it costs | – |
| [inf_matrix_sparsification](./inf_matrix_sparsification.ipynb) | Sparse vs. full influence matrix; truncation thresholds | – |

## IMRT and VMAT planning

| Notebook | What you learn | Needs |
|---|---|---|
| [vmat_scp_tutorial](./vmat_scp_tutorial.ipynb) | VMAT planning with column generation + sequential convex programming; export the DICOM-RT plan | – |
| [imrt_scp_tutorial](./imrt_scp_tutorial.ipynb) | Step-and-shoot IMRT with several apertures per beam via column generation + SCP | – |
| [dose_mimicking](./dose_mimicking.ipynb) | Optimize a plan to reproduce a given reference dose (dose mimicking) | Eclipse |

## How far from the global optimum? (mixed-integer programming references)

| Notebook | What you learn | Needs |
|---|---|---|
| [vmat_global_optimal](./vmat_global_optimal.ipynb) | Globally optimal VMAT plan on a down-sampled problem vs. column generation and SCP | MIP solver |
| [imrt_global_optimal](./imrt_global_optimal.ipynb) | Compares fluence-map + leaf sequencing, column generation, SCP and a deliverability-constrained MIP | Gurobi |
| [beam_orientation_global_optimal](./beam_orientation_global_optimal.ipynb) | Globally optimal beam angles for IMRT | MIP solver |
| [dvh_constraint_global_optimal](./dvh_constraint_global_optimal.ipynb) | Globally optimal plan that meets dose-volume-histogram constraints | Gurobi |

## AI-based dose prediction (`portpy.ai`)

| Notebook | What you learn | Needs |
|---|---|---|
| [beamlet_dose_prediction](./beamlet_dose_prediction.ipynb) | Task 1: predict the dose of individual beamlets with a 3D U-Net, assemble a predicted influence matrix, and plan with it | GPU, `portpy[ai]` |
| [imrt_dose_prediction](./imrt_dose_prediction.ipynb) | Task 2: predict a patient's 3D dose distribution and turn it into a deliverable IMRT plan | GPU, `portpy[ai]` |
| [vmat_scp_dose_prediction](./vmat_scp_dose_prediction.ipynb) | Task 2 for VMAT: predicted dose as the reference for SCP-based VMAT planning | GPU, `portpy[ai]` |

## Clinical integration

| Notebook | What you learn | Needs |
|---|---|---|
| [imrt_tps_import](./imrt_tps_import.ipynb) | Leaf-sequence an IMRT plan to DICOM-RT and import it into a TPS; export optimal fluence to Eclipse | TPS / Eclipse |
| [vmat_tps_import](./vmat_tps_import.ipynb) | Export a VMAT plan as DICOM-RT and import it into a TPS | TPS |
| [eclipse_photon_dose_calculation](./eclipse_photon_dose_calculation.ipynb) | Bring your own patients: extract beamlets, voxels and the influence matrix from Eclipse with Varian's [MAAS-DoseInfluenceMatrix](https://github.com/Varian-MedicalAffairsAppliedSolutions/MAAS-DoseInfluenceMatrix) tool, then plan in PortPy | Eclipse |
| [3d_slicer_integration](./3d_slicer_integration.ipynb) | View CT, structures and dose in 3D Slicer from a Jupyter notebook | 3D Slicer |
