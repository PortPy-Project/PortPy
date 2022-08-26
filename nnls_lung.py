from utils import *
from visualization import *

import numpy as np
import matplotlib.pyplot as plt

def main():
    patient_folder_path = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Lung_Patient_2'
    file_figure_path = r'C:\Users\fua\Pictures\Figures'
    file_figure_name = file_figure_path + r'\robust_smooth-lung_2.jpg'
    
    # Read all the meta data for the required patient
    meta_data = load_metadata(patient_folder_path)
    
    # Options for loading requested data
    # If 1, then load the data. If 0, then skip loading the data
    options = dict()
    # options['loadInfluenceMatrixFull'] = 0   # Skip loading full dose influence matrix.
    options['loadInfluenceMatrixFull'] = 1
    options['loadInfluenceMatrixSparse'] = 1
    options['loadBeamEyeViewStructureMask'] = 0

    # gantry_rtns = [8, 16, 24, 32, 80, 120]
    # coll_rtns = [0, 0, 0, 90, 0, 90]
    # beam_indices = [46,131,36,121,26,66,151,56,141]
    beam_indices = [10, 20, 30, 40]
    smooth_lambda = [0, 0.15, 0.25, 0.5, 1, 10]
    # smooth_lambda = [0, 0.15]
    # cutoff = 0.1
    cutoff = 0.01
    vol_perc = 0.9

    # Create IMRT Plan
    print("Creating IMRT plan...")
    my_plan = create_imrt_plan(meta_data, beam_indices = beam_indices, options = options)
    pres = my_plan['clinicalCriteria']['presPerFraction_Gy']
    i_ptv = get_voxels(my_plan, "PTV") - 1

    # print("Solving NNLS cutoff problem...")
    # w, obj, rob = run_nnls_optimization_cvx(my_plan, cutoff = 0.025, verbose = False)

    # print("\nSolving NNLS cutoff problem with smoothing...")
    # w_smooth, obj_smooth, rob_smooth = run_nnls_optimization_cvx(my_plan, cutoff = 0.025, lambda_x = 0.6, lambda_y = 0.4, verbose = False)
    
    print("Solving NNLS cutoff problem...")
    dose_true_diff = []
    for lam in smooth_lambda:
        print("Smoothing weight: {0}".format(lam))
        # w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = run_nnls_optimization_cvx(my_plan, cutoff = cutoff, lambda_x = lam, lambda_y = lam, sparse = True, verbose = False)
        w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = run_nnls_optimization_cvx(my_plan, cutoff = cutoff, lambda_x = lam, lambda_y = lam, sparse = False, verbose = False)
        
        # Scale dose vectors so V(90%) = p, i.e., 90% of PTV receives 100% of prescribed dose.
        d_true_smooth[i_ptv] = scale_dose(d_true_smooth[i_ptv], pres, vol_perc)
        d_cut_smooth[i_ptv] = scale_dose(d_cut_smooth[i_ptv], pres, vol_perc)
        
        rob_smooth = np.linalg.norm(d_cut_smooth - d_true_smooth)/np.linalg.norm(d_true_smooth)
        dose_true_diff.append(rob_smooth)
    
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    plt.plot(smooth_lambda, dose_true_diff)
    # plt.xlim(left = 0)
    # plt.ylim(bottom = 0)
    plt.xlabel("$\lambda$")
    # plt.ylabel("$||A^{cutoff}x - A^{true}x||_2$")
    plt.ylabel("$||A^{cutoff}x - A^{true}x||_2/||A^{true}x||_2$")
    plt.title("Robustness Error vs. Smoothing Weight (Lung Patient 2)")
    plt.show()
    
    fig.savefig(file_figure_name, bbox_inches = "tight", dpi = 300)

if __name__ == "__main__":
    main()
