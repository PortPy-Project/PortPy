# Changelog

All notable changes to PortPy are listed here. Full release notes are on the
[GitHub Releases](https://github.com/PortPy-Project/PortPy/releases) page.

## 1.2.1 — 2026-10

- New `portpy.photon.imrt_scp` module: step-and-shoot IMRT with several apertures per beam via
  column generation and sequential convex programming (`create_imrt_plan`,
  `ImrtOptimizationColGen`, `ImrtScpOptimization`); example notebook `imrt_scp_tutorial.ipynb`.
- New `imrt_global_optimal.ipynb`: deliverability-constrained MIP compared with leaf sequencing,
  column generation and SCP.
- Beamlet dose-prediction models reorganised; 3D ViT and Swin UNETR transformer baselines added.
- `Evaluation.get_plan_score` (overall plan score from a scorecard); scorecard path fix.
- Frozen benchmark data splits shipped in `portpy/config_files/data_splits/data_splits_default.json`
  and loadable with `DataExplorer.load_config_data_splits`.
- Example notebook updates and independent bug fixes.

## 1.2.0 — 2026-08-17

- Beamlet dose prediction pipeline in `portpy.ai`: preprocessing, dataset, training/testing and
  assembly of a predicted influence matrix (`beamlet_dose_prediction.ipynb`).
- New `portpy.ai.inference` subpackage: beam-frame geometry, ray-aligned BEV patches, GPU
  resampling, `RayUNetPredictor` and `PatientGridPredictor`.
- Two sampling grids for beamlet dose (`grid="patient" | "bev"`).
- Plan-quality scorecards: `Evaluation.compute_total_quality_score`, two shipped scorecards
  (Lung 60 Gy, Prostate SBRT) and `adapt_scorecard_prescription`.
- `UNet3D` model and an expanded `networks3d` registry; dose-prediction module made site-independent.
- VMAT SCP updated with bug fixes.

## 1.1.4 — 2026-03-17

- Prediction-based dose mimicking in VMAT column generation (PR #18).
- DVH-constraint and clinical-criteria bug fixes. Minimum Python raised to 3.11.

## 1.1.3 — 2026-01-12

- `download_portpy_data` script and CLI for selective downloads from Hugging Face.
- CVaR helper for DVH constraints in `Optimization` (PR #15).
- VMAT SCP MU scaling and clinical-criteria fixes (PR #14); VMAT template for Prostate 26 Fx.

## 1.1.2 — 2025-09-19

- Bug fixes in `portpy.ai`.

## 1.1.1 — 2025-09-10

- 129 prostate patients added to the Hugging Face dataset.
- `DataExplorer` methods for accessing data directly from Hugging Face.
- Unit tests and GitHub Actions CI/CD.
- Redundant max-dose constraints removed for faster optimization.

## 1.1.0 — 2025-05-15

- Dataset hosting moved to Hugging Face with in-library download; lung dataset expanded.
- VMAT column-generation warm start for SCP.
- Improved dose-distribution and clinical-criteria visualization; online data augmentation in `portpy.ai`.
- Hugging Face Space for browsing and downloading the dataset.

## 1.0.8 — 2025-03-31

- Photon dose calculation for custom patients outside the PortPy dataset (Eclipse users), with
  `eclipse_photon_dose_calculation.ipynb` and `create_ct_dose_voxel_map`.

## 1.0.6 — 2025-03-14

- Column generation as the initial solution for SCP-based VMAT; `pyproject.toml` build.

## 1.0.5 — 2025-01-13

- `portpy.ai` notebooks reworked; random leaf-position initialization in VMAT SCP.

## 1.0.4 — 2024-06-05 (patches through 1.0.4.6, 2024-11-08)

- SCP-based VMAT optimization module (`vmat_scp`) and DICOM-RT export of VMAT plans.
- VMAT dose-prediction pipeline and TPS-import notebooks.
- MOSEK and pydicom made optional extras; robust DVH plot; several bug fixes.

## 1.0.0 — 2023-09-29 (through 1.0.3, 2024-01-04)

- First stable release: 50 lung patients, clinical-criteria evaluation, CI/HI metrics,
  matured JSON configuration files.
