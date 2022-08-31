from utils import *
from visualization import *

import numpy as np
import matplotlib.pyplot as plt

def main():
    patient_num = 2
    patient_folder_path = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Lung_Patient_{0}'.format(patient_num)
    save_figure_path = r'C:\Users\fua\Pictures\Figures'
    save_data_path = r'C:\Users\fua\Documents\Data'
    save_figure_name = save_figure_path + r'\robust_smooth-lung_{0}.jpg'.format(patient_num)
    save_dvh_name = save_figure_path + r'\dvh-lung_{0}.jpg'.format(patient_num)
    fun_data_name = lambda s: save_data_path + r'\lung_{0}-{1}.npy'.format(patient_num, s)
    save_data = True
    
    # Read all the meta data for the required patient
    meta_data = load_metadata(patient_folder_path)
    
    # Options for loading requested data
    # If 1, then load the data. If 0, then skip loading the data
    options = dict()
    options['loadInfluenceMatrixFull'] = 1
    options['loadInfluenceMatrixSparse'] = 0
    options['loadBeamEyeViewStructureMask'] = 0

    # gantry_rtns = [8, 16, 24, 32, 80, 120]
    # coll_rtns = [0, 0, 0, 90, 0, 90]
    # beam_indices = [46, 131, 36, 121, 26, 66, 151, 56, 141]
    beam_indices = [10, 20, 30, 40]
    smooth_lambda = [0, 0.15, 0.25, 0.5, 1, 10]
    # smooth_lambda = [0, 0.5, 1]
    # cutoff = 0.1
    cutoff = 0.01
    vol_perc = 0.9

    # Create IMRT Plan
    print("Creating IMRT plan...")
    my_plan = create_imrt_plan(meta_data, beam_indices = beam_indices, options = options)
    pres = my_plan["clinicalCriteria"]["presPerFraction_Gy"]
    num_frac = my_plan["clinicalCriteria"]["numOfFraction"]
    i_ptv = get_voxels(my_plan, "PTV") - 1

    # print("Solving NNLS cutoff problem...")
    # w, obj, rob = run_nnls_optimization_cvx(my_plan, cutoff = 0.025, verbose = False)

    # print("\nSolving NNLS cutoff problem with smoothing...")
    # w_smooth, obj_smooth, rob_smooth = run_nnls_optimization_cvx(my_plan, cutoff = 0.025, lambda_x = 0.6, lambda_y = 0.4, verbose = False)
    
    print("Solving NNLS cutoff problem...")
    dose_true_raw = []
    dose_cut_raw = []
    
    dose_true_opt = []
    dose_cut_opt = []
    dose_diff_norm = []
    for lam in smooth_lambda:
        print("Smoothing weight: {0}".format(lam))
        # w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = run_nnls_optimization_cvx(my_plan, cutoff = cutoff, lambda_x = lam, lambda_y = lam, sparse = True, verbose = False)
        w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = run_nnls_optimization_cvx(my_plan, cutoff = cutoff, lambda_x = lam, lambda_y = lam, sparse = False, verbose = False)
        
        dose_true_raw.append(d_true_smooth)
        dose_cut_raw.append(d_cut_smooth)
        
        # Scale dose vectors so V(90%) = p, i.e., 90% of PTV receives 100% of prescribed dose.
        d_true_smooth = scale_dose(d_true_smooth, pres, vol_perc, i_ptv)
        d_cut_smooth = scale_dose(d_cut_smooth, pres, vol_perc, i_ptv)
        
        dose_true_opt.append(d_true_smooth)
        dose_cut_opt.append(d_cut_smooth)
        
        rob_smooth = np.linalg.norm(d_cut_smooth - d_true_smooth)/np.linalg.norm(d_true_smooth)
        dose_diff_norm.append(rob_smooth)
    
    # Save smoothing weights and dose matrices.
    if save_data:
        np.save(fun_data_name("lambda"), np.array(smooth_lambda))
        np.save(fun_data_name("dose_true"), np.column_stack(dose_true_raw))
        np.save(fun_data_name("dose_cut"), np.column_stack(dose_cut_raw))
        # np.save(fun_data_name("dose_true"), np.column_stack(dose_true_opt))
        # np.save(fun_data_name("dose_cut"), np.column_stack(dose_cut_opt))
    
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    plt.plot(smooth_lambda, dose_diff_norm)
    # plt.xlim(left = 0)
    # plt.ylim(bottom = 0)
    plt.xlabel("$\lambda$")
    # plt.ylabel("$||A^{cutoff}x - A^{true}x||_2$")
    plt.ylabel("$||A^{cutoff}x - A^{true}x||_2/||A^{true}x||_2$")
    plt.title("Robustness Error vs. Smoothing Weight (Lung Patient {0})".format(patient_num))
    plt.show()
    
    fig.savefig(save_figure_name, bbox_inches = "tight", dpi = 300)
    
    # Plot DVH curves.
    lam_idx = 1   # lambda = 0.5 or 1.0 seems to work best.
    orgs = ['PTV', 'GTV', 'LUNG_L', 'LUNG_R', 'ESOPHAGUS', 'HEART', 'CORD']
    plot_dvh(dose_true_opt[lam_idx], my_plan, orgs = orgs, title = "DVH for Lung Patient {0} ($\lambda$ = {1})".format(patient_num, smooth_lambda[lam_idx]), filename = save_dvh_name)

if __name__ == "__main__":
    main()
