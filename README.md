<p align="center">
  <img src="./images/PortPy_logo.png" width="40%" height="40%">
</p>

# What is PortPy?

PortPy, short for **P**lanning and **O**ptimization for **R**adiation **T**herapy, represents a collective effort to establish an open-source Python library dedicated to advancing the development and clinical implementation of cancer radiotherapy treatment planning algorithms. This initiative encompasses planning methodologies for Intensity Modulated Radiation Therapy (IMRT), Volumetric Modulated Arc Therapy (VMAT), along with other emerging modalities. PortPy provides clinical-grade data and coding resources that foster *benchmarking*, *reproducibility*, and *community development*.


**Contents**
- [What can you do with PortPy?](#WhatDo)
- [Quick Start](#QuickStart)
- [How to contribute?](#limitations)
- [The limitations of current version](#limitations)
- [Data](#Data)
- [Installation](#Installation)
- [Team](#Team)




# What can you do with PortPy? <a name="WhatDo"></a>
PortPy facilitates the **design**, **testing**, and **clinical validation** of your treatment planning algorithms. This includes both cutting-edge AI-based models and traditional optimization techniques. PortPy provides:
1. **Benchmark Dataset**
     * Access to data required for optimization, extracted directly from the FDA-approved Eclipse treatment planning system via its API 
     * A current set of data from 50 lung patients, which will be expanded to 100 lung patients by the end of 2023
     * A **benchmark IMRT plan** for each patient, created using our in-house automated planning system ([Edelman Award Finalist YouTube Video](https://youtu.be/895M6j5KjPs), [Paper](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.13572))
   
2. **Benchmark Algorithms:** Offering globally optimal solutions for:
     * Dose Volume Histogram (DVH) constraints (see [dvh_constraint_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/dvh_constraint_optimization.ipynb))
     * IMRT Beam Orientation Optimization (BOO) (see [beam_orientation_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/beam_orientation_optimization.ipynb))
     * Volumetric Modulated Arc Therapy (VMAT) (see [vmat_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/vmat_optimization.ipynb))
4. **Visualization**
     * Basic built-in visualization tools (e.g., DVH, dose distribution) are integrated into PortPy (see (see [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb)))
     * Enhanced visualizations are available through the integration with the popular open-source [3DSlicer](https://www.slicer.org/) package (see [3d_slicer_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/3d_slicer_integration.ipynb))
5. **Evaluation**  
     * PortPy IMRT plans with optimal fluence can be imported into Eclipse for leaf sequencing and final clinical evaluations  (see [eclipse_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/eclipse_integration.ipynb))
     * PortPy IMRT/VMAT plans can be imported into FDA approved TPS( e.g. Eclipse, Raystation) using DICOM RT Plan file for final clinical evaluations  (see [TPS-integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/TPS-Integration.ipynb))
     * Plans can also be evaluated within PortPy using well-established clinical protocols (e.g., Lung 2Gyx30, see  [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb))
     * Future updates will include more standardized RTOG metrics and outcome models (TCP/NTCP)
6. **Optimization** 
     * PortPy provides high-level optimization problem formulation and access to both free and commercial optimization engines through the integration with a popular open-source [CVXPy](https://www.cvxpy.org/) package (see [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb))
     * Commercial engines (e.g., [MOSEK](https://www.mosek.com/), [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer), [GUROBI](https://www.gurobi.com/)) are also free for academic and research use
7. **AI-Based Planning** 
     * The PortPy.AI module provides a framework for exploring AI-driven treatment planning  (see [dose_prediction_pipeline.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/AI_tutorials/dose_prediction_pipeline.ipynb))
     * The newly added PortPy.AI module includes a tutorial on predicting a 3D-dose distribution and converting the prediction into a deliverable plan

# Quick Start <a name="QuickStart"></a>

1. To grasp the primary features of PortPy, we highly recommend exploring the [basic_tutorial.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/1_basic_tutorial.ipynb) notebook
2. To understand how to import a PortPy plan into Eclipse for final evaluations, browse through the [eclipse_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/eclipse_integration.ipynb) notebook
3. To create dicom RT Plan file from PortPy plan and import into FDA approved commercial TPS (e.g. Eclipse, Raystation), please browse through [TPS-integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/TPS-Integration.ipynb) notebook
4. To learn about enhanced visualization techniques using the 3D-Slicer package, refer to the  [3d_slicer_integration.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/3d_slicer_integration.ipynb) notebook
5. For algorithm benchmarking, the global optimal solutions are provided for non-convex optimization problems resulting from beam angle optimization [beam_orientation_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/beam_orientation_optimization.ipynb), incorporating DVH constraints [dvh_constraint_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/dvh_constraint_optimization.ipynb), and VMAT optimization [vmat_optimization.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/benchmark_algorithms/vmat_optimization.ipynb) using the mixed-integer programming on down-sampled data.
6. If you encounter computational challenges with large-scale optimization problems, you can opt for down-sampling the voxels/beamlets, as illustrated in the [down_sampling](https://github.com/PortPy-Project/PortPy/blob/master/examples/advanced_tutorials/down_sampling.ipynb) notebook, or further sparsify the influence matrix, as demonstrated in the [inf_matrix_sparsification](https://github.com/PortPy-Project/PortPy/blob/master/examples/advanced_tutorials/inf_matrix_sparsification.ipynb) notebook.
7. To learn about creating a final deliverable plan using AI based model, please refer to [dose_prediction_pipeline.ipynb](https://github.com/PortPy-Project/PortPy/blob/master/examples/AI_tutorials/dose_prediction_pipeline.ipynb) notebook.

# How to contribute? <a name="HowContribute"></a>
To maintain the lightweight nature and user-friendliness of PortPy modules, our aim is to include only fundamental functionalities, along with benchmark data and algorithms. We will establish separate repositories within the [PortPy-Project orgainization](https://github.com/PortPy-Project) for projects developed by our team using PortPy as a platform. This is similar to what we've done for projects like [LowDimRT](https://github.com/PortPy-Project/LowDimRT) and [VMAT](https://github.com/PortPy-Project/ECHO-VMAT).

If you're interested in contributing to existing PortPy modules or wish to create a new module, we encourage you to contact us first. This will help ensure that our objectives and priorities are aligned. If you use PortPy to build your own package, you're welcome to host your package within the [PortPy-Project orgainization](https://github.com/PortPy-Project). Alternatively, you can host your package on your own GitHub page. In this case, please inform us so that we can fork it and feature it under the PortPy-Project organization.


# The limitations of current version of PortPy <a name="limitations"></a>
Current version of PortPy has the following limitations which would be addressed in the future updates:

1. You can only work with the benchmark dataset provided in this PortPy repo and cannot use your own dataset for now.
2. PortPy.Photon and PortPy.AI are the only modules available. You cannot do proton research with the current version. Please use [OpenTPS](http://opentps.org/) for proton research.

# Data <a name="Data"></a>

PortPy equips researchers with a robust benchmark patient dataset, sourced from the FDA-approved Eclipse commercial treatment planning system through its API. This dataset embodies all necessary elements for optimizing various machine configurations such as beam angles, aperture shapes, and leaf movements. It includes

1. **Dose Influence Matrix:** The dose contribution of each beamlet to each voxel,
2. **Beamlets/Voxels Details:** Detailed information about the position and size of beamlets/voxels,
3. **Expert-Selected Benchmark Beams:** An expert clinical physicist has carefully selected benchmark beams, providing reference beams for comparison and benchmarking,
4. **Benchmark IMRT Plan:** A benchmark IMRT plan generated using our in-house automated treatment planning system called ECHO ([Edelman Award Finalist YouTube Video](https://youtu.be/895M6j5KjPs), [Paper](https://aapm.onlinelibrary.wiley.com/doi/epdf/10.1002/mp.13572)). This plan serves as a benchmark for evaluating new treatment planning algorithms.

To access these resources, users are advised to download the latest version of the dataset, which can be found [here](https://drive.google.com/drive/folders/1nA1oHEhlmh2Hk8an9e0Oi0ye6LRPREit?usp=sharing). Subsequently, create a directory titled './data' in the current project directory and transfer the downloaded file into it. For example, ./data/Lung_Phantom_Patient_1.



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
| [Sadegh Alam](https://scholar.google.com/citations?user=iy7TlU0AAAAJ&hl=en)  | Adaptive Treatment Planning and Imaging          | Cornell     |
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
