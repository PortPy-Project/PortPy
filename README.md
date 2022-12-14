<p align="center">
  <img src="./images/PortPy_logo.jpg" width="50%">
</p>

# What is PortPy?
**Note: The package is at its early stages of development (version 0.xx) and we are now collecting feedback from reserachers to further refine the data structure and the main functionality. We are expecting to have a stable version 1.xx around March 2023. We would love to hear your feedback.**

PortPy (**P**lanning and **O**ptimization for **R**adiation **T**herapy) is a community effort to develop the **first opensource python library** to facilitate the development and clinical translation of radiotherapy cancer treatment planning algorithms. PortPy includes:
1. Research-ready data and code for *benchmarking*, *reproducibility*, and *community-driven* development.
2. Interface to an open-source optimization package [CVXPy](https://www.cvxpy.org/) for easy/quick prototyping and out-of-the-box access to commercial/opensource optimization engines (e.g., Mosek, Gorubi, CPLEX, IPOPT).
3. Visualization modules to visualize relevant plan information (e.g, dose volume histograms, dose distribution, fluence map).
4. Evaluation modules to quantify plan quality with respect to established clinical metrics (e.g., RTOG metrics, dose conformality, tumor control probability, normal tissue control probability).
# Data
Data needed for optimization and algorithm development (e.g., a set of beams/beamlets/voxels, dose contribution of each beamlet to each voxel) are provided for a set of pre-specified machine parameters (e.g., beam/collimator/couch angles). We will initially provide these for a set of publicly available datasets from [TCIA](https://www.cancerimagingarchive.net/). We hope to expand our dataset in the future. The data needed for optimization is extracted from the research version of Eclipse<sup>TM</sup> treatment planning system ([Varian Medical Systems](https://www.varian.com/)) using its API. 

You can download the sample patient data [here](https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit?usp=sharing).

Create a directory named 'Data' in the current project directory and copy the downloaded file to it, e.g ./Data/Lung_Patient_1


# Installing PortPy

- Clone this repository:
  ```bash
  git clone https://github.com/PortPy-Project/PortPy.git
  cd portpy
  ```

- You need to install the dependencies in either a python virtual environment or anaconda environment. Instructions for setting up in python virtual environment are as follows:

  Install all the dependencies present in requirements.txt:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  (venv) pip install -r requirements.txt
  ```

To better understand the PortPy functionality, we recommend running an example script eg_1_basics.py for creating and visualizing a sample IMRT plan.

# License
PortPy code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes.

# Team
PortPy is a community project initiated at [Memorial Sloan Kettering Cancer Center](https://www.mskcc.org/). It is currently developed and maintained by:
1. [Masoud Zarepisheh](https://masoudzp.github.io/) ([Memorial Sloan Kettering Cancer Center](https://www.mskcc.org/))
2. [Saad Nadeem](https://nadeemlab.org/) ([Memorial Sloan Kettering Cancer Center](https://www.mskcc.org/))
3. [Gourav Jhanwar](https://github.com/gourav3017) ([Memorial Sloan Kettering Cancer Center](https://www.mskcc.org/))
4. [Mojtaba Tefagh](https://github.com/mtefagh) ([Sharif University of Technology](https://en.sharif.edu/))
5. [Vicki Taasti](https://scholar.google.com/citations?user=PEPyvewAAAAJ&hl=en) ([Maastro Clinic, Netherlands](https://www.mskcc.org/))
6. [Seppo Tuomaala](https://www.linkedin.com/in/seppo-tuomaala-5b57913/) ([Varian Medical Systems](https://www.varian.com/))

