<p align="center">
  <img src="./images/PortPy_logo.png" width="40%" height="40%">
</p>

![Version](https://img.shields.io/static/v1?label=latest&message=v1.0.3&color=darkgreen)
[![Total Downloads](https://static.pepy.tech/personalized-badge/portpy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=total%20downloads)](https://pepy.tech/project/portpy?&left_text=totalusers)

# What is PortPy?

PortPy, short for **P**lanning and **O**ptimization for **R**adiation **T**herapy, represents a collective effort 
to establish an open-source Python library dedicated to advancing the development and clinical implementation of 
cancer radiotherapy treatment planning algorithms. This initiative encompasses planning methodologies for 
Intensity Modulated Radiation Therapy (IMRT), Volumetric Modulated Arc Therapy (VMAT), along with other 
emerging modalities. PortPy provides clinical-grade data and coding resources that foster *benchmarking*, 
*reproducibility*, and *community development*.

**Note:** In the near future, we plan to launch an educational YouTube channel to assist 
researchers new to this field. Meanwhile, we suggest reviewing relevant literature review papers
([Zarepisheh et al. 2021](https://pubsonline.informs.org/doi/abs/10.1287/inte.2021.1095), 
[Breedveld et al. 2019](https://www.sciencedirect.com/science/article/abs/pii/S0377221718307148), 
[Ehrgott et al. 2010](https://link.springer.com/article/10.1007/s10479-009-0659-4)) and watching YouTube videos 
([Edelman competition](https://www.youtube.com/watch?v=895M6j5KjPs&t=1025s), [Varian IMRT](https://www.youtube.com/watch?v=eZS6DVGBx0k), [Elekta VMAT](https://www.youtube.com/watch?v=AE1SxvnFT3s)).
  

**Contents**
- [Why we created PortPy?](#Why)
- [What can you do with PortPy?](#WhatDo)
- [Quick Start](#QuickStart)
- [How to contribute?](#limitations)
- [Data](#Data)
- [Installation](#Installation)
- [Team](#Team)

# Why we created PortPy? <a name="Why"></a>

<p align="center">
  <img src="./images/PortPy-Figure-12.png" width="90%" height="40%">
<p>


A key limitation of existing open-source packages in radiotherapy treatment planning is their
inability to objectively evaluate new treatment planning techniques against current clinical practices. 
This limitation stems from the difficulty in fully replicating clinical environments, such as detailed 
linear accelerator configurations and commercial dose calculation engines. Consequently,
plans generated using novel techniques, such as AI-based automated treatment planning, cannot be objectively compared
with clinical plans used in patient treatment. To address this limitation, we are integrating PortPy with 
commercial treatment planning systems (TPSs). PortPy is already compatible with the Varian Eclipse TPS, 
and we plan to expand compatibility to additional TPSs in the future. It's crucial to note that PortPy can 
be used independently of any TPS, but accessing a TPS is essential for final evaluation within the TPS (see left figure above).

Right figure above highlights the inspiration for developing PortPy, drawing on successful open-source practices 
from the AI and computer science communities. Tools like PyTorch and TensorFlow, along with benchmark datasets 
such as ImageNet and algorithms like AlexNet, have transformed AI and data science. Our aim is to emulate 
this success, providing researchers with comprehensive resources (PortPy integration with commercial TPS 
and other renowned open-source tools such as 3DSlicer and CVXPy), a benchmark dataset 
(featuring 50 curated lung patients with expertly selected beams, all necessary optimization data, 
and benchmark IMRT plans created using our in-house automated planning system, ECHO), 
and benchmark algorithms, including Mixed Integer Programming (MIP) algorithms for achieving globally optimal solutions.




# What can you do with PortPy? <a name="WhatDo"></a>

<p align="center">
<img src="./images/PortPy-Workflow.png" width="90%" height="40%">
<p>

PortPy facilitates the **design**, **testing**, and **clinical validation** of new treatment planning algorithms. 
This includes both cutting-edge AI-based models and traditional optimization techniques. 
The above figure illustrates the PortPy design and its three main
modules: “Data Management”, “Plan Generation”, and “Plan
Evaluation”,  which are discussed below. We recommend reviewing our Jupyter Notebooks 
examples for a more comprehensive understanding of these modules. 

1. **Data Management**
     * This module provides access to the curated benchmark PortPy dataset, which currently comprises data from 50 lung patients.
It allows researchers to test and develop their algorithms using the same dataset (see [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb) notebook)
     * The available data includes: 1) CT images and contours, 2) all necessary data for optimization extracted from 
Eclipse using its API (version 16.1), 3)expert-selected beams for each patient, 4) an IMRT plan for each patient, 
generated using our in-house automated planning system, 
ECHO ([YouTube Video](https://youtu.be/895M6j5KjPs), [Paper](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.13572)).
More information about data can be found in [Data](#Data) section.
     * In the current version, you can only work with the benchmark dataset provided in this PortPy repo and 
   cannot use your own dataset for now. We will address this problem in the near future

2. **Plan Generation**
     * This module facilitates the generation of treatment plans using either classical optimization methods or 
emerging AI-based techniques 
     * For optimization tasks, PortPy has been integrated with [CVXPy](https://www.cvxpy.org/), a widely-used open-source package. 
CVXPy enables the high-level formulation of optimization problems and offers out-of-the-box access to a range
of free (e.g., [SCIP](https://www.scipopt.org/), [SCIPY](https://docs.scipy.org/doc/scipy/reference/optimize.html)) and commercial (e.g., [MOSEK](https://www.mosek.com/), [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer), [GUROBI](https://www.gurobi.com/)) optimization engines (available for free for research
purposes) (see [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb) notebook)
     * PortPy.AI module is equipped with essential functionalities for AI-based planning. These include data access, 
data pre-processing, model training and testing, and patient-specific 3D dose prediction 
(see [dose_prediction_pipeline.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/AI_tutorials/dose_prediction_pipeline.ipynb) notebook)
     * The availability of both optimization and AI-based planning modules within PortPy allows researchers 
to not only compare these techniques but also explore their complementary aspects

3. **Plan Visualization and Evaluation**
     * Basic built-in visualization tools (e.g., DVH, dose distribution) are integrated into PortPy 
     * Enhanced visualizations are available through the integration with the popular open-source [3DSlicer](https://www.slicer.org/) package (see [3d_slicer_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/3d_slicer_integration.ipynb) notebook)
     * PortPy IMRT plans with optimal fluence can be imported into Eclipse for leaf sequencing and final clinical evaluations  (see [eclipse_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/eclipse_integration.ipynb))
     * PortPy IMRT/VMAT plans with optimal control points can be imported into any FDA approved TPS ( e.g. Eclipse, Raystation) using DICOM RT Plan file for final clinical evaluations  (see [TPS-integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/TPS-Integration.ipynb)).
It should be noted that some discrepancies are anticipated 
with TPSs other than Eclipse, due to the fact that our current data are derived from Eclipse, and there 
may exist variations in dose calculation methods between Eclipse and other TPSs. 
     * Plans can also be evaluated within PortPy using well-established clinical protocols (e.g., Lung 2Gyx30, see  [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb))
     * Future updates will include more standardized RTOG metrics and outcome models (TCP/NTCP)



# Quick Start <a name="QuickStart"></a>

1. To grasp the primary features of PortPy, we highly recommend exploring the [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb) notebook
2. To understand how to import a PortPy plan into Eclipse for final evaluations, browse through the [eclipse_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/eclipse_integration.ipynb) notebook
3. To create dicom RT Plan file from PortPy plan and import into FDA approved commercial TPS (e.g. Eclipse, Raystation), please browse through [TPS-integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/TPS-Integration.ipynb) notebook
4. To learn about enhanced visualization techniques using the 3D-Slicer package, refer to the  [3d_slicer_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/3d_slicer_integration.ipynb) notebook
5. For algorithm benchmarking, the global optimal solutions are provided for non-convex optimization problems resulting from beam angle optimization [beam_orientation_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/beam_orientation_optimization.ipynb), incorporating DVH constraints [dvh_constraint_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/dvh_constraint_optimization.ipynb), and VMAT optimization [vmat_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/vmat_optimization.ipynb) using the mixed-integer programming on down-sampled data.
6. If you encounter computational challenges with large-scale optimization problems, you can opt for down-sampling the voxels/beamlets, as illustrated in the [down_sampling](https://github.com/PortPy-Project/PortPy/blob/master/examples/advanced_tutorials/down_sampling.ipynb) notebook, or further sparsify the influence matrix, as demonstrated in the [inf_matrix_sparsification](https://github.com/PortPy-Project/PortPy/blob/master/examples/advanced_tutorials/inf_matrix_sparsification.ipynb) notebook.
7. To learn about creating a final deliverable plan using AI based model, please refer to [dose_prediction_pipeline.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/AI_tutorials/dose_prediction_pipeline.ipynb) notebook.

# How to contribute? <a name="HowContribute"></a>
<p align="center">
<img src="./images/PortPy-Figs_3.jpg" width="90%" height="40%">
<p>
As illustrated in the above figure, PortPy organization includes "PortPy", which is the current repository, 
and PortPy extensions, which are the repositories developed using the PortPy as a platform. 
To maintain the lightweight nature and user-friendliness of PortPy modules, our aim is to include only fundamental 
functionalities, along with benchmark data and algorithms in the PortPy repo, and establish separate repositories 
for other projects, similar to what we've done for projects like [LowDimRT](https://github.com/PortPy-Project/LowDimRT)
and [ECHO VMAT](https://github.com/PortPy-Project/ECHO-VMAT).

If you're interested in contributing to existing PortPy modules or wish to create a new module, 
we encourage you to contact us first. This will help ensure that our objectives and priorities are aligned. 
If you use PortPy to build your own package, you're welcome to host your package within the
[PortPy-Project orgainization](https://github.com/PortPy-Project). 
Alternatively, you can host your package on your own GitHub page. In this case, 
please inform us so that we can fork it and feature it under the PortPy-Project organization. 
For those keen on creating a logo for their repository, we offer the option to customize our [pre-designed logo]().


# Data <a name="Data"></a>
<p align="center">
<img src="./images/PortPy-Figs_4_data.jpg" width="90%" height="40%">
<p>
PortPy equips researchers with a robust benchmark patient dataset, sourced from the FDA-approved Eclipse commercial treatment planning system through its API. This dataset embodies all necessary elements for optimizing various machine configurations such as beam angles, aperture shapes, and leaf movements. It includes

1. **Dose Influence Matrix (AKA dose deposition matrix, dij matrix):** The dose contribution of each beamlet to each voxel,
2. **Beamlets/Voxels Details:** Detailed information about the position and size of beamlets/voxels,
3. **Expert-Selected Benchmark Beams:** An expert clinical physicist has carefully selected benchmark beams, providing reference beams for comparison and benchmarking,
4. **Benchmark IMRT Plan:** A benchmark IMRT plan generated using our in-house automated treatment planning system called ECHO ([YouTube Video](https://youtu.be/895M6j5KjPs), [Paper](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.13572)). This plan serves as a benchmark for evaluating new treatment planning algorithms.
5. **Benchmark Clinical Criteria:** A set of clinically relevant mean/max/DVH criteria for plan evaluation. 
Currently, this set encompasses only the Lung 2Gy×30 protocol but will be expanded in the future to more protocols as well as TCP/NTCP evaluation functions.  

To access these resources, users are advised to download the latest version of the dataset, 
which can be found [here](https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit?usp=sharing). 
Subsequently, create a directory titled './data' in the current project directory and transfer the downloaded 
file into it. For example, ./data/Lung_Phantom_Patient_1. 
We have adopted the widely-used JSON and HDF5 formats for data storage.
[HDFViwer](https://www.hdfgroup.org/downloads/hdfview/) can be utilized to view the contents of the HDF5 files.  



**Note:** Initially, we will utilize a lung dataset from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics). The original DICOM CT images and structure sets are not included in the PortPy dataset and need to be directly downloaded from the TCIA. Users can fetch the **TCIA collection ID** and the **TCIA subject ID** for each PortPy patient using the *get_tcia_metadata()* method in PortPy and subsequently download the data from TCIA (see [eclipse_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/eclipse_integration.ipynb))


# Installation <a name="Installation"></a>

1. Install using pip:
   
    * Run the command '**pip install portpy**'

2. Install using conda:

    * Run the command '**conda install -c conda-forge portpy**'

3. Install from source:
   
    * Clone this repository using '**git clone https://github.com/PortPy-Project/PortPy.git**'
    * Navigate to the repository with '**cd portpy**'

    * Install the dependencies within a Python virtual environment or Anaconda environment. To set up in a Python virtual environment, install all the dependencies specified in requirements.txt as follows:
        * Create the virtual environment with '**python3 -m venv venv**'
        * Activate the environment with '**source venv/bin/activate**'
        * Install the requirements using '**(venv) pip install -r requirements.txt**'


# Team <a name="Team"></a>
PortPy is a community project initiated at [Memorial Sloan Kettering Cancer Center](https://www.mskcc.org/). It is currently developed and maintained by:

| Name                                                                         | Expertise                                        | Institution |
|------------------------------------------------------------------------------|--------------------------------------------------|-------------|
| [Masoud Zarepisheh](https://masoudzp.github.io/)                             | Treatment Planning and Optimization              | MSK         |
| [Saad Nadeem](https://nadeemlab.org/)                                        | Computer Vision and AI in Medical Imaging        | MSK         |
| [Gourav Jhanwar](https://github.com/gourav3017)                              | Algorithm Design and Development                 | MSK         |
| [Mojtaba Tefagh](https://github.com/mtefagh)                                 | Mathematical Modeling and Reinforcement Learning | SUT         |
| [Vicki Taasti](https://scholar.google.com/citations?user=PEPyvewAAAAJ&hl=en) | Physics and Planning of Proton Therapy           | MAASTRO     |
| [Seppo Tuomaala](https://www.linkedin.com/in/seppo-tuomaala-5b57913/)        | Eclispe API Scripting                            | VARIAN      |

# License <a name="License"></a>
PortPy code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes.

# Reference <a name="Reference"></a>
If you find our work useful in your research or if you use parts of this code please cite our [AAPM'23 abstract](https://aapm.confex.com/aapm/2023am/meetingapp.cgi/Paper/4208) :
```
@article{jhanwar2023portpy,
  title={Portpy: An Open-Source Python Package for Planning and Optimization in Radiation Therapy Including Benchmark Data and Algorithms},
  author={Jhanwar, Gourav and Tefagh, Mojtaba and Taasti, Vicki T and Alam, Sadegh R and Tuomaala, Seppo and Nadeem, Saad and Zarepisheh, Masoud},
  journal={AAPM 65th Annual Meeting & Exhibition},
  year={2023}
}
```
