# Winter Of Code Projects for PortPy

Working on AI/ML involves many different tools from data visualizations, model/algorithm development, evaluation and testing. Below is the list of projects that can help you to gain hands on experience with these tools.
## Index
### 1. Visualization Tools  
  1.1 [Interactive and User-Friendly Visualization for Dose-Volume Histograms (DVH)](#dvh)  
  1.2 [Enhancing Dose Distribution Analysis through User-Friendly Visualization Tools](#dose)  
  1.3 [User-Friendly Interactive Dashboard for PortPy Data Visualization with Dynamic Tables](#dashboard)  
  1.4 [Interactive Visualization of Multi-Leaf Collimator (MLC) Movements for Treatment Plans](#mlc)

### 2. Website Development
  2.1 [Creating a Simple Website for PortPy](#website)

### 3. AI/ML advanced modelling
  3.1 [Implementing Top-Performing Dose Prediction Model (Cascade 3D U-Net)](#openkbp1)  
  3.2 [Building Dose Prediction using Advanced U-Net Architecture (Dilated U-Net)](#openkbp2)  
  3.3 [Feature-Based Dose Prediction with Enhanced Accuracy](#openkbp3)  

### 4. Optimization Algorithms
  4.1 [Gradient-Based Optimization for Radiotherapy](#gradient)  
  4.2 [Multi-GPU-Based ADMM for Large-Scale Radiotherapy Optimization](#admm)  

<br/>

---
## 1. Visualization Tools
<h3 id="dvh"> 1.1 Project Title: Interactive and User-Friendly Visualization for Dose-Volume Histograms (DVH) </h3>

**Requirements:** Matplotlib, Plotly (or alternative)  
**Good to Know:** Dash, PyQt (or alternative)  

**Short description:**

_What is DVH?_

A DVH is a convenient two-dimensional (2D) plot for plan evaluation and can be easily calculated from the 3D delivered dose. It shows how much of a structure (volume) gets a specific radiation dose (Gy) .The x-axis shows the dose (in Gray). The y-axis shows the percentage of the volume of the structure receiving that dose.

PortPy currently does visualization using Matplotlib. We would like to create an interactive visualization using Plotly (or an alternative). We would also like the visualization to work in both Jupyter Notebook and  Desktop versions

<img src="../images/dvh-example.png" alt="DVH plot" width="60%" height="40%">

**Project Outcome:**  
- Interactive DVH line plots.
- Ability to select structures through an interactive checkbox.
- Build, visualize, and interpret DVHs using provided data.

---

<br/>

<h3 id="dose"> 1.2 Project Title: Enhancing Dose Distribution Analysis through User-Friendly Visualization Tools </h3>

**Requirements:** Matplotlib, Plotly (or alternative)  
**Good to Know:** Dash, PyQt (or alternative)  

**Short description:**

_What is dose distribution?_

A spatial map that shows how radiation dose is spread across the patient body and can be visualized as 2d grid. It consist of slice of CT image, structure contours and dose as color wash on top of CT with some opacity. This project would create interactive 2d slice views of patient geometry with dose distribution

PortPy currently does visualization using Matplotlib. We would like to create an interactive visualization using Plotly (or an alternative). We would also like the visualization to work in both Jupyter Notebook and  Desktop versions

<img src="../images/dose_distribution.png" alt="Dose distribution" width="60%" height="40%">

**Project Outcome:**  
- View and explore dose maps interactively.
- Overlay structure contours on dose maps.
- Gain hands-on experience with medical imaging visualization.


<br/>

---

<h3 id="dashboard"> 1.3 Project Title: User-Friendly Interactive Dashboard for PortPy Data Visualization with Dynamic Tables </h3>

**Requirements:** Matplotlib, Plotly (or alternative)  
**Good to Know:** Dash, PyQt (or alternative)  

**Short description:**

_What does PortPy Data consist of?_

PortPy consist of two types of data i.e. **metadata** which are small size json files and **data** which are huge size **hdf5** files. This project will help users to visualize the metadata which is in **hierarchical** format as interactive table

![Dashboard Example](../images/dashboard.png)
**Metadata** is displayed as a static Pandas DataFrame, which limits interactivity and exploration. We would like to create an interactive dashboard using Plotly (or an alternative). We would also like the visualization to work in both Jupyter Notebook and  Desktop versions

**Project Outcome:**  
- Transform static metadata tables into dynamic, interactive dashboards.
- Enable search, filter, and expand capabilities for metadata exploration.


<br/>

---

<h3 id="MLC"> 1.4 Project Title: Interactive Visualization of Multi-Leaf Collimator (MLC) Movements for Treatment Plans </h3>

**Requirements:** Matplotlib, Plotly (or alternative)  
**Good to Know:** Dash, PyQt (or alternative)  

**Short description:**

_What is MLC?_

An **MLC** is a key component in radiotherapy machines used to shape radiation beams. It consists of movable metal leaves that block parts of the beam, ensuring the dose conforms to the tumor shape while sparing healthy tissue.

Current state: MLC movements are displayed as static visualizations, which limit interactivity and analysis. We would like to create an interactive visualization using Plotly (or an alternative). We would also like the visualization to work in both Jupyter Notebook and  Desktop versions
<img src="../images/MLC.png" alt="MLC" width="60%" height="40%">

**Project Outcome:**  
- Animate MLC movements across beam angles.
- Overlay targets and structures for better analysis.

<br/>

---

## 2. Website Development
<h3 id="website"> 2.1 Project Title: Creating a Simple Website for PortPy </h3>

**Requirements:** Markdown, reStructuredText, Sphinx  
**Good to Know:** HTML, CSS, JavaScript  

**Short description:**
Design and build a simple website for PortPy using modern web technologies.

![Website Example](../images/web_1.png)
![Website Example](../images/web_2.png)
![Website Example](../images/web_3.png)
![Website Example](../images/web_4.png)
![Website Example](../images/web_5.png)

**Project Outcome:**  
- Create a visually appealing and user-friendly website.
- Gain hands-on experience with web development.

<br/>

---


## 3. AI/ML advanced modelling


<h3 id="openkbp1"> 3.1 Project Title: Implementing Top-Performing Dose Prediction Model (Cascade 3D U-Net) from the Open-Access Grand Challenge </h3>

**Requirements:** Proficiency in PyTorch, GPU-based training  
**Good to Know:** TensorFlow, Keras, medical imaging concepts  

**Short description:**

This project focuses on integrating the winning Cascade 3D U-Net model from the open-access grand challenge into the PortPy framework. The task involves training and evaluating the model on the PortPy dataset, ensuring optimal performance for radiotherapy dose prediction. The implementation should align with PortPy's standards for usability and extensibility, contributing to advanced dose prediction workflows.

**Project Outcome:**  
- Train and implement a state-of-the-art dose prediction model.
- Enhance workflows for radiotherapy dose prediction.


<br/>

---

<h3 id="openkbp2"> 3.2 Project Title: Build dose prediction using advanced U-Net architecture Dilated U-Net (Runner-Up Model for Open-Access Grand Challenge) </h3>

**Requirements:** Proficiency in PyTorch, GPU-based training  
**Good to Know:** TensorFlow, Keras, medical imaging concepts  

**Short description:**

This project focuses on integrating the runner up Dilated U-Net model from the open-access grand challenge into the PortPy framework. The task involves training and evaluating the model on the PortPy dataset, ensuring optimal performance for radiotherapy dose prediction. The implementation should align with PortPy's standards for usability and extensibility, contributing to advanced dose prediction workflows.

**Project Outcome:**  
- Develop an advanced dose prediction model.
- Gain experience in radiotherapy and AI techniques.

<br/>

---

<h3 id="openkbp3"> 3.3 Project Title: Implement Feature-Based Dose Prediction (Runner-Up Model): Use feature-based losses and One Cycle Learning for enhanced accuracy </h3>


**Requirements:** Proficiency in PyTorch, GPU-based training  
**Good to Know:** TensorFlow, Keras, medical imaging concepts  

**Short description:**

This project focuses on integrating the runner model based on feature-based losses and one cycle learning from the open-access grand challenge into the PortPy.AI framework. The task involves training and evaluating the model on the PortPy dataset, ensuring optimal performance for radiotherapy dose prediction. The implementation should align with PortPy's standards for usability and extensibility, contributing to advanced dose prediction workflows.

**Project Outcome:**  
- Implement and evaluate feature-based dose prediction models.
- Enhance accuracy in radiotherapy dose predictions.


<br/>

---

## 4. Optimization Algorithms
<h3 id="gradient"> 4.1 Project Title: Gradient-Based Optimization: Use JAX Auto-Diff for Radiotherapy Optimization with Gradient Descent </h3>

**Requirements:** Proficiency in Python, familiarity with JAX  
**Good to Know:** Optimization techniques, NumPy, SciPy  

**Short description:**

This project aims to leverage JAX's automatic differentiation capabilities to implement gradient-based optimization for radiotherapy treatment planning. The focus will be on developing efficient optimization pipelines using gradient descent.

**Project Outcome:**  
- Develop a scalable and efficient framework for gradient-based radiotherapy optimization.
- Implement gradient descent using JAX for dose optimization tasks.
- Demonstrate improvements in computation time and plan quality through gradient-based techniques.
- Provide insights into the impact of gradient-based methods on clinical treatment planning workflows.
- Deliver a reproducible Python-based implementation integrated into the PortPy framework.

<br/>

---

<h3 id="admm"> 4.2 Project Title: Implement Scalable and Fast ADMM Algorithms for Large-Scale Radiotherapy Optimization </h3>

**Requirements:** Proficiency in Python, ADMM 
**Good to Know:** JAX, CUDA, PyTorch, optimization techniques  

**Short description:**

This project focuses on developing scalable and efficient ADMM (Alternating Direction Method of Multipliers) algorithms tailored for large-scale radiotherapy optimization problems. Utilizing multi-GPU systems, the implementation will achieve faster convergence and enhanced scalability, enabling solutions to complex radiotherapy optimization tasks.

**Project Outcome:**  
- Demonstrate scalability and speed improvements for large-scale radiotherapy optimization problems.
- Benchmark the ADMM implementation against existing optimization methods.
- Deliver a Python-based implementation compatible with existing radiotherapy frameworks, such as PortPy.
- Gain insights into the role of parallelized ADMM in solving real-world optimization challenges in radiotherapy

<br/>
