from utils import *
from visualization import *

import numpy as np
import matplotlib.pyplot as plt

def main():
    patient_folder_path = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Paraspinal_Patient_2'
    save_figure_path = r'C:\Users\fua\Pictures\Figures'
    save_data_path = r'C:\Users\fua\Documents\Data'
    save_figure_name = save_figure_path + r'\robust_smooth-paraspinal.jpg'
    save_dvh_nom_name = save_figure_path + r'\dvh_nominal-paraspinal.jpg'
    save_dvh_band_name = save_figure_path + r'\dvh_bands-paraspinal.jpg'
    fun_data_name = lambda s: save_data_path + r'\paraspinal_2-{0}.npy'.format(s)
    save_data = True
    
    # Read all the metadata for the required patient.
    meta_data = load_metadata(patient_folder_path)
    
    # Skip loading full dose influence matrix.
    options = dict()
    options['loadInfluenceMatrixFull'] = 0
    options['loadInfluenceMatrixSparse'] = 1
    options['loadBeamEyeViewStructureMask'] = 0
    
    # Beams for different patient movements +/- (x,y,z) direction.
    # Gantry angles: 220, 240, 260, 100, 120, 140, 160, 180
    beam_indices_nom = np.arange(1,10)
    beam_indices_move = {"Xmin3":   np.arange(10,19),
                         "Xplus3":  np.arange(19,28),
                         "Ymin3":   np.arange(28,37),
                         "Yplus3":  np.arange(37,46),
                         "Zmin3":   np.arange(46,55),
                         "Zplus3":  np.arange(55,64)
                        }

    # Optimization parameters.
    # smooth_lambda = [0, 0.15, 0.25, 0.5, 1, 10]
    smooth_lambda = [0, 0.01, 0.05, 0.15, 0.25, 0.5]
    # smooth_lambda = [0, 0.5]
    vol_perc = 0.9
    
    # Create IMRT plan for nominal scenario.
    print("Creating IMRT plan for nominal scenario...")
    my_plan_nom = create_imrt_plan(meta_data, beam_indices = beam_indices_nom, options = options)
    pres = my_plan_nom["clinicalCriteria"]["presPerFraction_Gy"]
    num_frac = my_plan_nom["clinicalCriteria"]["numOfFraction"]
    i_ptv = get_voxels(my_plan_nom, "PTV") - 1
    
    print("Solving NNLS nominal problem...")
    beam_opt_true = []
    dose_opt_true = []
    dose_opt_raw = []
    for lam in smooth_lambda:
        print("Smoothing weight: {0}".format(lam))
        w_smooth, obj_smooth, d_true_smooth, d_cut_smooth = run_nnls_optimization_cvx(my_plan_nom, cutoff = 0, lambda_x = lam, lambda_y = lam, verbose = False)
        dose_opt_raw.append(d_true_smooth)
        
        # Scale dose vectors so V(90%) = p, i.e., 90% of PTV receives 100% of prescribed dose.
        d_true_smooth = scale_dose(d_true_smooth, pres, vol_perc, i_ptv)
        # d_cut_smooth = scale_dose(d_cut_smooth, pres, vol_perc, i_ptv)
        
        beam_opt_true.append(w_smooth)
        dose_opt_true.append(d_true_smooth)
    
    # Form matrices with rows = beamlets intensities/dose voxels, columns = smoothing weights.
    beam_opt_true_mat = np.column_stack(beam_opt_true)
    dose_opt_true_mat = np.column_stack(dose_opt_true)
    dose_opt_true_norm = np.linalg.norm(dose_opt_true_mat, axis = 0)   # ||A^{nom}*x_l||_2 for l = 1,...,len(smooth_lambda).
    
    # Save smoothing weights and nominal dose matrix.
    if save_data:
        np.save(fun_data_name("lambda"), smooth_lambda)
        np.save(fun_data_name("dose_true"), np.column_stack(dose_opt_raw))
        # np.save(fun_data_name("dose_true"), dose_opt_true)

    # Compute deviation of dose in movement scenarios from nominal dose.
    print("Computing robustness error...")
    dose_diff_mat_norm = []
    dose_move_mat_list = []
    for scenario_name, beam_indices in beam_indices_move.items():
        # Import dose-influence matrix for movement scenario.
        print("Movement scenario: {0}".format(scenario_name))
        my_plan_move = create_imrt_plan(meta_data, beam_indices = beam_indices, options = options)
        
        inf_mat_move = my_plan_move["infMatrixSparse"]
        pres = my_plan_move["clinicalCriteria"]["presPerFraction_Gy"]
        i_ptv = get_voxels(my_plan_move, "PTV") - 1
        
        # Compute dose delivered by optimal beamlets.
        dose_move_mat = inf_mat_move @ beam_opt_true_mat              # Rows = dose voxels, columns = smoothing weights.
        
        # Save dose matrix for scenario.
        if save_data:
            np.save(fun_data_name("dose_move-{0}".format(scenario_name)), dose_move_mat)
        
        # For each scenario and smoothing weight, scale dose vector so V(90%) = p, i.e., 90% of PTV receives 100% of prescribed dose.
        for j in range(len(smooth_lambda)):
            dose_move_mat[:,j] = scale_dose(dose_move_mat[:,j], pres, vol_perc, i_ptv)
        dose_move_mat_list.append(dose_move_mat)
        
        dose_diff_mat = dose_move_mat - dose_opt_true_mat             # Column l = A^{s}*x_l - A^{nom}*x_l, where s = movement scenario.
        dose_diff_norm = np.linalg.norm(dose_diff_mat, axis = 0)      # ||A^{s}*x_l - A^{nom}*x_l||_2 for l = 1,...,len(smoothLambda).
        dose_diff_mat_norm.append(dose_diff_norm)
    
    # Form matrix with rows = scenarios, columns = smoothing weights.
    dose_diff_mat_norm = np.row_stack(dose_diff_mat_norm)
    dose_diff_sum = np.sum(dose_diff_mat_norm, axis = 0)
    dose_diff_sum_norm = dose_diff_sum/dose_opt_true_norm
   
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    plt.plot(smooth_lambda, dose_diff_sum_norm)
    # plt.semilogy(smooth_lambda, dose_diff_sum_norm)
    plt.xlabel("$\lambda$")
    plt.ylabel("$\sum_{s=1}^N ||A^{s}x_{\lambda} - A^{nom}x_{\lambda}||_2/\sum_{s=1}^N ||A^{nom}x_{\lambda}||_2$")
    plt.title("Robustness Error vs. Smoothing Weight (Paraspinal Patient 2)")
    plt.show()
    
    fig.savefig(save_figure_name, bbox_inches = "tight", dpi = 300)
    
    # Plot DVH curves for nominal scenario.
    lam_idx = 2   # lambda = 0.01 seems to work best.
    orgs = ['PTV', 'CTV', 'LUNG_L', 'LUNG_R', 'ESOPHAGUS', 'HEART', 'CORD']
    plot_dvh(dose_opt_true_mat[:,lam_idx], my_plan_nom, orgs = orgs, title = "DVH for Paraspinal Nominal Scenario ($\lambda$ = {0})".format(smooth_lambda[lam_idx]), filename = save_dvh_nom_name)

    # Plot DVH bands for nominal and movement scenarios.
    dose_list = [dose_opt_true_mat[:,lam_idx]] + [dose_move_mat[:,lam_idx] for dose_move_mat in dose_move_mat_list]
    plot_robust_dvh(dose_list, my_plan_nom, orgs = orgs, title = "DVH Bands for all Paraspinal Scenarios ($\lambda$ = {0})".format(smooth_lambda[lam_idx]), filename = save_dvh_band_name)

if __name__ == "__main__":
    main()
