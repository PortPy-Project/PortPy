from utils import *
from visualization import *
from evaluation import get_dose

import numpy as np
import matplotlib.pyplot as plt

def create_lung_plot(patient_num = 1, show_fig = True, save_fig = True):
    patient_folder_path = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Lung_Patient_{0}'.format(patient_num)
    save_data_path = r'C:\Users\fua\Documents\Data'
    fun_data_name = lambda s: save_data_path + r'\lung_{0}-{1}.npy'.format(patient_num, s)
    
    save_figure_path = r'C:\Users\fua\Pictures\Figures'
    save_figure_name = save_figure_path + r'\robust_smooth-lung_{0}.jpg'.format(patient_num) if save_fig else None
    save_dvh_name = save_figure_path + r'\dvh-lung_{0}.jpg'.format(patient_num) if save_fig else None
    
    beam_indices = [10, 20, 30, 40]
    orgs = ["PTV", "ESOPHAGUS", "HEART", "LUNG_L", "LUNG_R"]
    # cutoff = 0.01
    # vol_perc = 0.9
    vol_norm = 90
    
    # Read all the meta data for the required patient
    meta_data = load_metadata(patient_folder_path)
    
    # Options for loading requested data
    # If 1, then load the data. If 0, then skip loading the data
    options = dict()
    options["loadInfluenceMatrixFull"] = 0
    options["loadInfluenceMatrixSparse"] = 0
    options["loadBeamEyeViewStructureMask"] = 0
    
    # Create IMRT Plan
    print("Creating IMRT plan...")
    my_plan = create_imrt_plan(meta_data, beam_indices = beam_indices, options = options)
    pres = my_plan["clinicalCriteria"]["presPerFraction_Gy"]
    num_frac = my_plan["clinicalCriteria"]["numOfFraction"]
    i_ptv = get_voxels(my_plan, "PTV") - 1
    
    # Load saved smoothing weights and dose matrices.
    print("Loading saved data...")
    smooth_lambda = np.load(fun_data_name("lambda"))
    d_true_mat = np.load(fun_data_name("dose_true"))
    d_cut_mat = np.load(fun_data_name("dose_cut"))
    
    print("Computing relative scaled difference...")
    d_true_scale_list = []
    d_cut_scale_list = []
    d_diff_norm = []
    for j in range(len(smooth_lambda)):
        # Scale dose vectors so V(90%) = p, i.e., vol_perc% of PTV receives 100% of prescribed dose.
        # d_true_perc = np.percentile(d_true_mat[i_ptv,j], 1 - vol_perc)
        # d_true_scale = (pres / d_true_perc)*d_true_mat[:,j]
        # d_true_scale = scale_dose(d_true_mat[:,j], pres, vol_perc, i_ptv)
        norm_factor = get_dose(d_true_mat[:,j], my_plan, "PTV", vol_norm)/pres
        d_true_scale = d_true_mat[:,j]/norm_factor
        d_true_scale_list.append(d_true_scale)
    
        # d_cut_perc = np.percentile(d_cut_mat[i_ptv,j], 1 - vol_perc)
        # d_cut_scale = (pres / d_cut_perc)*d_cut_mat[:,j]
        # d_cut_scale = scale_dose(d_cut_mat[:,j], pres, vol_perc, i_ptv)
        norm_factor = get_dose(d_cut_mat[:,j], my_plan, "PTV", vol_norm)/pres
        d_cut_scale = d_cut_mat[:,j]/norm_factor
        d_cut_scale_list.append(d_cut_scale)
        
        # Compute relative difference between scaled optimal doses.
        rob_smooth = np.linalg.norm(d_cut_scale - d_true_scale)/np.linalg.norm(d_true_scale)
        d_diff_norm.append(rob_smooth)
    
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    plt.plot(smooth_lambda, d_diff_norm)
    # plt.xlim(left = 0)
    # plt.ylim(bottom = 0)
    plt.xlabel("$\lambda$")
    # plt.ylabel("$||A^{cutoff}x - A^{true}x||_2$")
    plt.ylabel("$||A^{cutoff}x - A^{true}x||_2/||A^{true}x||_2$")
    plt.title("Lung Patient {0}: Robustness Error vs. Smoothing Weight".format(patient_num))
    
    if show_fig:
        plt.show()
    if save_fig:
        fig.savefig(save_figure_name, bbox_inches = "tight", dpi = 300)
    
    # Plot DVH curves.
    lam_idx = 1
    # lam_idx = np.argmin(d_diff_norm)   # Find smoothing weight that achieves lowest robustness error.
    plot_dvh(d_true_mat[:,lam_idx], my_plan, orgs = orgs, norm_flag = True, norm_volume = vol_norm, norm_struct = "PTV", 
             title = "Lung Patient {0}: DVH ($\lambda$ = {1})".format(patient_num, smooth_lambda[lam_idx]), show = show_fig, filename = save_dvh_name)

def create_paraspinal_plot(patient_num = 1, show_fig = True, save_fig = True):
    patient_folder_path = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Data\Paraspinal_Patient_{0}'.format(patient_num)
    save_data_path = r'C:\Users\fua\Documents\Data'
    fun_data_name = lambda s: save_data_path + r'\paraspinal_{0}-{1}.npy'.format(patient_num, s)
    
    save_figure_path = r'C:\Users\fua\Pictures\Figures'
    save_figure_name = save_figure_path + r'\robust_smooth-paraspinal_{0}.jpg'.format(patient_num) if save_fig else None
    save_dvh_nom_name = save_figure_path + r'\dvh_nominal-paraspinal_{0}.jpg'.format(patient_num) if save_fig else None
    save_dvh_band_name = save_figure_path + r'\dvh_bands-paraspinal_{0}.jpg'.format(patient_num) if save_fig else None
    
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
    orgs = ["PTV", "CORD", "ESOPHAGUS", "LUNG_L", "LUNG_R"]
    # vol_perc = 0.9
    vol_norm = 90
    
    # Read all the metadata for the required patient.
    meta_data = load_metadata(patient_folder_path)
    
    # Options for loading requested data
    # If 1, then load the data. If 0, then skip loading the data
    options = dict()
    options["loadInfluenceMatrixFull"] = 0
    options["loadInfluenceMatrixSparse"] = 0
    options["loadBeamEyeViewStructureMask"] = 0
    
    # Create IMRT plan for nominal scenario.
    print("Creating IMRT plan for nominal scenario...")
    my_plan_nom = create_imrt_plan(meta_data, beam_indices = beam_indices_nom, options = options)
    pres = my_plan_nom["clinicalCriteria"]["presPerFraction_Gy"]
    num_frac = my_plan_nom["clinicalCriteria"]["numOfFraction"]
    i_ptv = get_voxels(my_plan_nom, "PTV") - 1
    
    # Load saved smoothing weights and dose matrices.
    print("Loading saved data...")
    smooth_lambda = np.load(fun_data_name("lambda"))
    dose_nom_mat = np.load(fun_data_name("dose_true"))
    dose_move_mat_list = [np.load(fun_data_name("dose_move-{0}".format(scenario))) for scenario in beam_indices_move.keys()]
    
    # Scale dose vectors so V(90%) = p, i.e., vol_perc% of PTV receives 100% of prescribed dose.
    print("Scaling dose vectors...")
    print("Nominal scenario")
    dose_nom_scale_mat = np.zeros(dose_nom_mat.shape)
    for j in range(len(smooth_lambda)):
        # dose_nom_mat[:,j] = scale_dose(dose_nom_mat[:,j], pres, vol_perc, i_ptv)
        norm_factor = get_dose(dose_nom_mat[:,j], my_plan_nom, "PTV", vol_norm)/pres
        dose_nom_scale_mat[:,j] = dose_nom_mat[:,j]/norm_factor
    
    k = 0
    dose_move_scale_mat_list = []
    for scenario, beam_indices in beam_indices_move.items():
        print("Movement scenario: {0}".format(scenario))
        my_plan_move = create_imrt_plan(meta_data, beam_indices = beam_indices, options = options)
        
        dose_move_scale = np.zeros(dose_move_mat_list[k].shape)
        for j in range(len(smooth_lambda)):
            dose_move_vec = dose_move_mat_list[k][:,j]
            norm_factor = get_dose(dose_move_vec, my_plan_move, "PTV", vol_norm)/pres
            dose_move_scale[:,j] = dose_move_vec/norm_factor
        dose_move_scale_mat_list.append(dose_move_scale)
        k = k + 1
            
    print("Computing relative scaled difference...")
    dose_diff_mat_norm = []
    for k in range(len(dose_move_mat_list)):
        dose_diff_mat = dose_move_scale_mat_list[k] - dose_nom_scale_mat   # Column l = A^{s}*x_l - A^{nom}*x_l, where s = movement scenario.
        dose_diff_norm = np.linalg.norm(dose_diff_mat, axis = 0)           # ||A^{s}*x_l - A^{nom}*x_l||_2 for l = 1,...,len(smoothLambda).
        dose_diff_mat_norm.append(dose_diff_norm)
    
    # Form matrix with rows = scenarios, columns = smoothing weights.
    dose_diff_mat_norm = np.row_stack(dose_diff_mat_norm)
    dose_diff_sum = np.sum(dose_diff_mat_norm, axis = 0)
    dose_nom_scale_norm = np.linalg.norm(dose_nom_scale_mat, axis = 0)     # ||A^{nom}*x_l||_2 for l = 1,...,len(smooth_lambda).
    dose_diff_sum_norm = dose_diff_sum/dose_nom_scale_norm
    
    # Plot robustness measure versus smoothing weight.
    fig = plt.figure(figsize = (12,8))
    plt.plot(smooth_lambda, dose_diff_sum_norm)
    # plt.semilogy(smooth_lambda, dose_diff_sum_norm)
    plt.xlabel("$\lambda$")
    plt.ylabel("$\sum_{s=1}^N ||A^{s}x_{\lambda} - A^{nom}x_{\lambda}||_2/||A^{nom}x_{\lambda}||_2$")
    plt.title("Paraspinal Patient {0}: Robustness Error vs. Smoothing Weight".format(patient_num))
    
    if show_fig:
        plt.show()
    if save_fig:
        fig.savefig(save_figure_name, bbox_inches = "tight", dpi = 300)
    
    # Plot DVH curves for nominal scenario.
    lam_idx = 2
    # lam_idx = np.argmin(dose_diff_sum_norm)   # Find smoothing weight that achieves lowest robustness error.
    plot_dvh(dose_nom_mat[:,lam_idx], my_plan_nom, orgs = orgs, norm_flag = True, norm_volume = vol_norm, norm_struct = "PTV", 
             title = "Paraspinal Patient {0}: DVH for Nominal Scenario ($\lambda$ = {1})".format(patient_num, smooth_lambda[lam_idx]), show = show_fig, filename = save_dvh_nom_name)

    # Plot DVH bands for nominal and movement scenarios.
    dose_list = [dose_nom_mat[:,lam_idx]] + [dose_move_mat[:,lam_idx] for dose_move_mat in dose_move_mat_list]
    plot_robust_dvh(dose_list, my_plan_nom, orgs = orgs, norm_flag = True, norm_volume = vol_norm, norm_struct = "PTV", 
                    title = "Paraspinal Patient {0}: DVH Bands for all Scenarios ($\lambda$ = {1})".format(patient_num, smooth_lambda[lam_idx]), show = show_fig, filename = save_dvh_band_name)
    
if __name__ == "__main__":
    create_lung_plot(1, show_fig = True, save_fig = True)
    create_lung_plot(2, show_fig = True, save_fig = True)
    create_paraspinal_plot(2, show_fig = True, save_fig = True)
