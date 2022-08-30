from utils import *
from visualization import *

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
    orgs = ['PTV', 'GTV', 'LUNG_L', 'LUNG_R', 'ESOPHAGUS', 'HEART', 'CORD']
    # cutoff = 0.01
    vol_perc = 0.9
    
    # Read all the meta data for the required patient
    meta_data = load_metadata(patient_folder_path)
    
    # Options for loading requested data
    # If 1, then load the data. If 0, then skip loading the data
    options = dict()
    options['loadInfluenceMatrixFull'] = 0
    options['loadInfluenceMatrixSparse'] = 0
    options['loadBeamEyeViewStructureMask'] = 0
    
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
        d_true_perc = np.percentile(d_true_mat[i_ptv,j], 1 - vol_perc)
        d_true_scale = (pres / d_true_perc)*d_true_mat[:,j]
        d_true_scale_list.append(d_true_scale)
    
        d_cut_perc = np.percentile(d_cut_mat[i_ptv,j], 1 - vol_perc)
        d_cut_scale = (pres / d_cut_perc)*d_cut_mat[:,j]
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
    plt.title("Robustness Error vs. Smoothing Weight for Lung Patient {0}".format(patient_num))
    
    if show_fig:
        plt.show()
    if save_fig:
        fig.savefig(save_figure_name, bbox_inches = "tight", dpi = 300)
    
    # Plot DVH curves.
    lam_idx = 1   # lambda = 0.5 or 1.0 seems to work best.
    plot_dvh(d_true_scale_list[lam_idx], my_plan, orgs = orgs, title = "DVH for Lung Patient {0} ($\lambda$ = {1})".format(patient_num, smooth_lambda[lam_idx]), 
             show = show_fig, filename = save_dvh_name)
    
if __name__ == "__main__":
    create_lung_plot(1, show_fig = True, save_fig = True)
    create_lung_plot(2, show_fig = True, save_fig = True)
