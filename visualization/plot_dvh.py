import matplotlib.pyplot as plt
import random
from utils import get_voxels
import numpy as np


def get_colors(num):
    random.seed(42)
    colors = []
    for i in range(num):
        color = (random.random(), random.random(), random.random())
        colors.append(color)

    return colors


def plot_dvh(my_plan, dose, orgs=None, style='solid', norm_flag=0, weight_flag=1, width=None, colors=None, figsize=(12, 8), legend_font_size=10, title=None, filename=None, show=True, *args, **kwargs):

    plt.rcParams['font.size'] = 12
    if width is None:
        if style == 'dotted' or style == 'dashed':
            width = 2.5
        else:
            width = 2
    if colors is None:
        colors = get_colors(30)
    if orgs is None:
        orgs = []
        orgs = my_plan['structures']['Names']
    max_dose = 0.0
    all_orgs = my_plan['structures']['Names']
    # orgs = [org.upper for org in orgs]
    legend = []
    fig = plt.figure(figsize=figsize)
    for i in range(np.size(all_orgs)):
        if all_orgs[i] not in orgs:
            continue
        vox = get_voxels(my_plan, all_orgs[i])
        org_sort_dose = np.sort(dose[vox - 1])
        max_dose = np.maximum(max_dose, org_sort_dose[-1])
        sort_ind = np.argsort(dose[vox - 1])
        org_sort_dose = np.append(org_sort_dose, org_sort_dose[-1] + 0.01)
        x = org_sort_dose
        if weight_flag:
            org_points_spacing = my_plan['optVoxels']['VoxelSpacing_XYZ_mm'][0]
            org_points_volume = org_points_spacing[:, 0] * org_points_spacing[:, 1] * org_points_spacing[:, 2]
            org_points_sort_spacing = my_plan['optVoxels']['VoxelSpacing_XYZ_mm'][0][sort_ind]
            org_points_sort_volume = org_points_sort_spacing[:, 0] * org_points_sort_spacing[:,
                                                                     1] * org_points_sort_spacing[:, 2]
            sum_weight = np.sum(org_points_sort_volume)
            y = [1]
            for j in range(len(org_points_sort_volume)):
                y.append(y[-1] - org_points_sort_volume[j] / sum_weight)
        else:
            y = np.ones(len(vox) + 1) - np.arange(0, len(vox)+1) / len(vox)
        y[-1] = 0
        y = np.array(y)
        plt.plot(x, 100*y, linestyle=style, linewidth=width, color=colors[i], *args, **kwargs)
        legend.append(all_orgs[i])

    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume Fraction (%)')
    plt.xlim(0, max_dose*1.1)
    plt.ylim(0, 100)
    plt.legend(legend, prop={'size': legend_font_size}, loc="upper right")
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
    y = np.arange(0, 101)
    if norm_flag:
        x = my_plan['clinicalCriteria']['presPerFraction_Gy']*my_plan['clinicalCriteria']['numOfFraction']*np.ones_like(y)
    else:
        x = my_plan['clinicalCriteria']['presPerFraction_Gy']*np.ones_like(y)
    plt.plot(x, y, color='black')
    if title:
        plt.title(title)
    if show:
        plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=300)
