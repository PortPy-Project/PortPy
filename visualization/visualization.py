import matplotlib.pyplot as plt
import random
import numpy as np
# from evaluation import get_dose
# from utils.plan import Plan
from evaluation.evaluation import Evaluation
from .surface_plot import surface_plot


class Visualization(Evaluation):

    def __init__(self):
        super().__init__()
        self.beams = None
        self.structures = None
        self.clinical_criteria = None
        self.optimal_intensity = None

    def plot_robust_dvh(self, dose_list, my_plan, orgs=None, style='solid', norm_flag=False, norm_volume=90,
                        norm_struct='PTV', weight_flag=True, plot_scenario=None, width=None, colors=None,
                        figsize=(12, 8), legend_font_size=10, title=None, filename=None, show=True, *args, **kwargs):
        if not isinstance(dose_list, list):
            dose_list = [dose_list]
        if len(dose_list) == 0:
            raise ValueError("dose_list is empty")
        if width is None:
            if style == 'dotted' or style == 'dashed':
                width = 2.5
            else:
                width = 2
        if colors is None:
            colors = self.get_colors(30)
        if orgs is None:
            # orgs = []
            orgs = self.structures.structures_dict['name']
        max_dose = 0.0
        all_orgs = self.structures.structures_dict['name']
        # orgs = [org.upper for org in orgs]
        pres = self.clinical_criteria['pres_per_fraction_gy'] * self.clinical_criteria[
            'num_of_fractions']
        legend = []
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.size'] = 12
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in orgs:
                continue
            dose_sort_list = []
            for dose in dose_list:
                x, y = self.get_dvh(dose, all_orgs[i], weight_flag=weight_flag)
                dose_sort_list.append(x)
            d_sort_mat = np.column_stack(dose_sort_list)
            # Compute min/max DVH curve taken across scenarios.
            d_min_mat = np.min(d_sort_mat, axis=1)
            d_max_mat = np.max(d_sort_mat, axis=1)

            # Plot user-specified scenarios.
            if plot_scenario is not None:
                if not isinstance(dose_list, list):
                    plot_scenario = [plot_scenario]

                for n in range(len(plot_scenario)):
                    scene_num = plot_scenario[n]
                    if norm_flag:
                        norm_factor = self.get_dose(dose_list[scene_num], my_plan, norm_struct, norm_volume) / pres
                        dose_sort_list[scene_num] = dose_sort_list[scene_num] / norm_factor
                        d_min_mat = d_min_mat / norm_factor
                        d_max_mat = d_max_mat / norm_factor
                    plt.plot(dose_sort_list[scene_num], 100 * y, linestyle=style, color=colors[i], linewidth=width,
                             *args, **kwargs)
            max_dose = np.maximum(max_dose, d_max_mat[-1])
            plt.plot(d_min_mat, 100 * y, linestyle='dotted', linewidth=width, color=colors[i], *args, **kwargs)
            plt.plot(d_max_mat, 100 * y, linestyle='dotted', linewidth=width, color=colors[i], *args, **kwargs)
            plt.fill_betweenx(100 * y, d_min_mat, d_max_mat, alpha=0.25, label=all_orgs[i], color=colors[i], *args,
                              **kwargs)
            legend.append(all_orgs[i])

        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume Fraction (%)')
        plt.xlim(0, max_dose * 1.1)
        plt.ylim(0, 100)
        plt.legend(prop={'size': legend_font_size}, loc="upper right")
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        y = np.arange(0, 101)
        if norm_flag:
            x = self.clinical_criteria['pres_per_fraction_gy'] * self.clinical_criteria[
                'num_of_fractions'] * np.ones_like(y)
        else:
            x = self.clinical_criteria['pres_per_fraction_gy'] * np.ones_like(y)
        plt.plot(x, y, color='black')
        if title:
            plt.title(title)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)

    def plot_dvh(self, dose=None, orgs=None, style='solid', norm_flag=False, norm_volume=90, norm_struct='PTV',
                 weight_flag=True, width=None, colors=None,
                 figsize=(12, 8), legend_font_size=10, title=None, filename=None, show=True, *args, **kwargs):

        if dose is None:
            dose = self.beams.get_influence_matrix() * self.beams.optimal_intensity
        plt.rcParams['font.size'] = 12
        if width is None:
            if style == 'dotted' or style == 'dashed':
                width = 2.5
            else:
                width = 2
        if colors is None:
            colors = self.get_colors(30)
        if orgs is None:
            # orgs = []
            orgs = self.structures.structures_dict['name']
        max_dose = 0.0
        all_orgs = self.structures.structures_dict['name']
        # orgs = [org.upper for org in orgs]
        pres = self.clinical_criteria['pres_per_fraction_gy'] * self.clinical_criteria[
            'num_of_fractions']
        legend = []
        fig = plt.figure(figsize=figsize)
        if norm_flag:
            norm_factor = self.get_dose(self, dose, norm_struct, norm_volume) / pres
            dose = dose / norm_factor
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in orgs:
                continue
            # for dose in dose_list:
            #
            x, y = self.get_dvh(dose, all_orgs[i], weight_flag=weight_flag)
            max_dose = np.maximum(max_dose, x[-1])
            plt.plot(x, 100 * y, linestyle=style, linewidth=width, color=colors[i], *args, **kwargs)
            legend.append(all_orgs[i])

        plt.xlabel('Dose (Gy)')
        plt.ylabel('Volume Fraction (%)')
        plt.xlim(0, max_dose * 1.1)
        plt.ylim(0, 100)
        plt.legend(legend, prop={'size': legend_font_size}, loc="upper right")
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        y = np.arange(0, 101)
        if norm_flag:
            x = pres * np.ones_like(y)
        else:
            x = self.clinical_criteria['pres_per_fraction_gy'] * np.ones_like(y)
        plt.plot(x, y, color='black')
        if title:
            plt.title(title)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)

    @staticmethod
    def get_colors(num):
        random.seed(42)
        colors = []
        for i in range(num):
            color = (random.random(), random.random(), random.random())
            colors.append(color)

        return colors

    @staticmethod
    def surface_plot(matrix, **kwargs):
        return surface_plot(matrix, **kwargs)
