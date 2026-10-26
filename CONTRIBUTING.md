# Contributing to PortPy

Thank you for your interest in PortPy. Contributions of all kinds are welcome: bug
reports, documentation fixes, new example notebooks, new optimization or AI methods,
and integrations with other tools.

## Where things live

PortPy-Project is organised as a **core repository** plus **extension repositories**:

| Repository | Purpose |
|---|---|
| [PortPy](https://github.com/PortPy-Project/PortPy) (this repo) | Core library, benchmark data access, reference algorithms, example notebooks |
| [ECHO-VMAT](https://github.com/PortPy-Project/ECHO-VMAT) | Hierarchical (prioritized) automated VMAT planning |
| [CompressRTP](https://github.com/PortPy-Project/CompressRTP) | Influence-matrix and fluence compression for fast optimization |
| [portpy.github.io](https://github.com/PortPy-Project/portpy.github.io) | Documentation website |

To keep the core lightweight, project-specific research code belongs in an extension
repository. You can host it under the PortPy-Project organisation or on your own GitHub;
if you choose the latter, let us know and we will feature it.

## Before you start a larger change

Please open an [issue](https://github.com/PortPy-Project/PortPy/issues) or email the
lead developer (Gourav Jhanwar, jhanwarg@mskcc.org) describing what you plan to do. This
avoids duplicate effort and lets us agree on scope and interfaces first.

## Reporting bugs

Open an issue using the *Bug report* template. Include:

- PortPy version (`python -c "import portpy; print(portpy.__version__)"`), Python version, OS
- The patient id and protocol you used (if from the PortPy dataset)
- The solver (`MOSEK`, `CLARABEL`, `SCS`, `GUROBI`, ...)
- A minimal snippet and the full traceback

## Development setup

```bash
git clone https://github.com/PortPy-Project/PortPy.git
cd PortPy
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[test]"                               # add ",ai" for the deep-learning module
pytest
```

Notes:

- Python 3.11 or newer is required.
- The tests download `Lung_Phantom_Patient_1` from Hugging Face on first run and solve with
  the free SCS solver, so no commercial license is needed.
- The CI workflow (`.github/workflows/ci.yml`) runs the same `pytest` on Ubuntu / Python 3.12
  for every pull request that touches `portpy/`.

## Pull request guidelines

1. Fork the repository and create a branch from `master` (for example `feature/my-change`).
2. Keep pull requests focused. One feature or fix per PR is easiest to review.
3. Follow the existing code style: `snake_case` for functions and variables, `CamelCase`
   for classes, type hints and reStructuredText-style docstrings (`:param ...:`,
   `:return:`) as used throughout `portpy/photon/`.
4. Add or update tests under `tests/` when you change behaviour.
5. If you add a public method, make sure it is documented in its docstring; the website's
   API reference is generated from docstrings.
6. If you add or rename an example notebook, update the examples table in `README.md`
   and `examples/README.md`, and add a `.py` version under `examples/python_files/`.
7. Describe *what* and *why* in the PR description, and link the related issue.

## Example notebooks

New notebooks are very welcome. Please:

- Use a patient from the public dataset (prefer `Lung_Phantom_Patient_1` or a small
  down-sampled problem when the goal is to illustrate an idea quickly).
- State the requirements at the top (GPU, Eclipse TPS, 3D Slicer, commercial solver).
- Clear large outputs before committing so the repository stays small.

## Data contributions

The benchmark dataset is hosted on
[Hugging Face](https://huggingface.co/datasets/PortPy-Project/PortPy_Dataset). If you
would like to contribute patients (for example extracted from your Eclipse system with the
[Varian MAAS-DoseInfluenceMatrix](https://github.com/Varian-MedicalAffairsAppliedSolutions/MAAS-DoseInfluenceMatrix)
tool), please contact us first; data must be de-identified and shareable under a
compatible license.

## License

By contributing you agree that your contributions are licensed under the project's
license: Apache 2.0 with the Commons Clause (non-commercial). See `LICENSE.txt`.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful
and constructive.
