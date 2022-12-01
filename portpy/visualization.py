import time

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
import random
import numpy as np
# from evaluation import get_dose
# from portpy.plan import Plan
from portpy.evaluation import Evaluation

# from .surface_plot import surface_plot
# import SimpleITK as sitk
from portpy.beam import Beams
from portpy.structures import Structures
from portpy.clinical_criteria import ClinicalCriteria
# from ipywidgets.widgets import interact
from matplotlib.widgets import CheckButtons
from matplotlib.lines import Line2D
mpl.use('TkAgg')  # or can use 'Qt5Agg', whatever you have/prefer


class Visualization:
    #
    # def __init__(self):
    #     super().__init__()
    #     self.beams = None
    #     self.structures = None
    #     self.clinical_criteria = None
    #     self.optimal_intensity = None

    def __init__(self, beams: Beams, structures: Structures, clinical_criteria: ClinicalCriteria, evaluate: Evaluation,
                 ct=None):
        self._ct = ct
        self._structures = structures
        self._clinical_criteria = clinical_criteria
        self.beams = beams
        self._evaluate = evaluate
        self.dose_1d = None
        self.slider = None
        self.check = None

    def plot_robust_dvh(self, dose_list, my_plan, orgs=None, style='solid', norm_flag=False, norm_volume=90,
                        norm_struct='PTV', weight_flag=True, plot_scenario=None, width=None, colors=None,
                        figsize=(12, 8), legend_font_size=10, title=None, filename=None, show=False, *args, **kwargs):
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
            orgs = self._structures.structures_dict['name']
        max_dose = 0.0
        all_orgs = self._structures.structures_dict['name']
        # orgs = [org.upper for org in orgs]
        pres = self._clinical_criteria['pres_per_fraction_gy'] * self._clinical_criteria[
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
            x = self._clinical_criteria['pres_per_fraction_gy'] * self._clinical_criteria[
                'num_of_fractions'] * np.ones_like(y)
        else:
            x = self._clinical_criteria['pres_per_fraction_gy'] * np.ones_like(y)
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
            # dose = self.beams.get_influence_matrix() @ self.beams.optimal_intensity
            dose = self._structures.opt_voxels_dict['dose_1d']
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
            orgs = self._structures.structures_dict['name']
        max_dose = 0.0
        all_orgs = self._structures.structures_dict['name']
        # orgs = [org.upper for org in orgs]
        pres = self._clinical_criteria.clinical_criteria_dict['pres_per_fraction_gy'] * \
               self._clinical_criteria.clinical_criteria_dict[
                   'num_of_fractions']
        legend = []
        fig = plt.figure(figsize=figsize)
        if norm_flag:
            norm_factor = self._evaluate.get_dose(dose=dose, struct=norm_struct, volume_per=norm_volume) / pres
            dose = dose / norm_factor
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in orgs:
                continue
            # for dose in dose_list:
            #
            x, y = self._evaluate.get_dvh(dose=dose, struct=all_orgs[i], weight_flag=weight_flag)
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
            x = self._clinical_criteria.clinical_criteria_dict['pres_per_fraction_gy'] * np.ones_like(y)
        plt.plot(x, y, color='black')
        if title:
            plt.title(title)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)

    def plot_binary_mask_points(self, structure, show=True, color=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ind = self._structures.structures_dict['name'].index(structure)
        mask_3d = self._structures.structures_dict['structure_mask_3d'][ind]
        pos = np.where(mask_3d == 1)
        if color is None:
            color = self.get_colors(0)
        ax.scatter(pos[0], pos[1], pos[2], c=color)
        if show:
            plt.show()

    def plot_3d_CT(self):
        # slide through dicom images using a slide bar
        plt.figure(1)

        def dicom_animation(x):
            plt.imshow(self._ct['ct_hu_3D'][0][x, :, :])
            return x

        # interact(dicom_animation, x=(0, len(self._ct['ct_hu_3D'][0].shape[0]) - 1))

    # def plot_3d_dose(self, overlay_ct=True):
    #     # slide through dicom images using a slide bar
    #     mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
    #     plt.figure(1)
    #     plt.ion()
    #     dose_1d = self.beams.get_influence_matrix()*self.beams.optimal_intensity
    #     dose_vox_map = self._structures.opt_voxels_dict['ct_to_dose_voxel_map'][0]
    #     # dose_3d = np.zeros_like(dose_vox_map, dtype=float)
    #     # for ind in range(len(dose_1d)):
    #     #     if dose_1d[ind] > 0:
    #     #         dose_3d[np.where(dose_vox_map == dose_1d[ind])] = dose_1d[ind]
    #
    #     def dicom_animation(x):
    #         if overlay_ct:
    #             plt.imshow(self._ct['ct_hu_3D'][0][x, :, :], cmap='gray')
    #         dose_2d = np.zeros_like(dose_vox_map[x, :, :], dtype=float)
    #         inds = np.unique(dose_vox_map[x, :, :][dose_vox_map[x, :, :] > 0])
    #         for i in range(len(inds)):
    #             ind = inds[i]
    #             if dose_1d[ind] > 0:
    #                 dose_2d[np.where(dose_2d == dose_1d[ind])] = dose_1d[ind]
    #         # plt.imshow(dose_3d[x, :, :], alpha=0.6, cmap='rainbow')
    #         plt.imshow(dose_2d, alpha=0.4, cmap='rainbow')
    #         plt.title = 'Slice# {}'.format(x)
    #
    #         return x
    #
    #     interact(dicom_animation, x=(0, self._ct['ct_hu_3D'][0].shape[0] - 1))
    #     print('test')

    def plot_3d_dose_jupyter(self, overlay_ct=True):
        # slide through dicom images using a slide bar
        # mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
        plt.figure(1)
        dose_1d = self.beams.get_influence_matrix() * self.beams.optimal_intensity
        dose_vox_map = self._structures.opt_voxels_dict['ct_to_dose_voxel_map'][0]

        # dose_3d = np.zeros_like(dose_vox_map, dtype=float)
        # for ind in range(len(dose_1d)):
        #     if dose_1d[ind] > 0:
        #         dose_3d[np.where(dose_vox_map == dose_1d[ind])] = dose_1d[ind]

        def dicom_animation(x):
            if overlay_ct:
                plt.imshow(self._ct['ct_hu_3D'][0][x, :, :], cmap='gray')
            dose_2d = np.zeros_like(dose_vox_map[x, :, :], dtype=float)
            inds = np.unique(dose_vox_map[x, :, :][dose_vox_map[x, :, :] > 0])
            for i in range(len(inds)):
                ind = inds[i]
                if dose_1d[ind] > 0:
                    dose_2d[np.where(dose_vox_map[x, :, :] == ind)] = dose_1d[ind]
            # plt.imshow(dose_3d[x, :, :], alpha=0.6, cmap='rainbow')
            plt.imshow(dose_2d, alpha=0.4, cmap='rainbow')
            plt.title('Slice# {}'.format(x))

            return x

        # interact(dicom_animation, x=(0, self._ct['ct_hu_3D'][0].shape[0] - 1))
        print('test')

    @staticmethod
    def get_colors(num):
        random.seed(42)
        colors = []
        for i in range(num):
            color = (random.random(), random.random(), random.random())
            colors.append(color)

        return colors

    # @staticmethod
    # def surface_plot(matrix, **kwargs):
    #     return surface_plot(matrix, **kwargs)
    @staticmethod
    def surface_plot(matrix, **kwargs):
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, np.transpose(matrix), **kwargs)
        return fig, ax, surf

    def plot_fluence_2d(self, beam_id=None):
        self.beams.plot_fluence_2d(beam_id=beam_id)

    def plot_fluence_3d(self, beam_id=None):
        self.beams.plot_fluence_3d(beam_id=beam_id)

    @staticmethod
    def remove_keymap_conflicts(new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def plot_3d_dose(self, dpi=100, show_dose=True, show_struct=True, show_isodose=True):
        # Visualization.remove_keymap_conflicts({'left', 'right'})
        # ct_volume = self._ct['ct_hu_3D'][0]
        fig, ax = plt.subplots(dpi=dpi)
        plt.rcParams["figure.autolayout"] = True
        self.fig = fig
        self.ax = ax
        self.show_dose = show_dose
        self.show_struct = show_struct
        self.show_isodose = show_isodose
        self.volume = self._ct['ct_hu_3D'][0]
        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.2, bottom=0.2)

        ax.index = 40
        if show_dose:
            dose_3d = self.create_3d_dose()
            self.dose_3d = dose_3d
            ax.imshow(self.volume[ax.index, :, :], cmap='gray')
            im = ax.imshow(dose_3d[ax.index, :, :], alpha=0.4 * (dose_3d[ax.index, :, :] > 0), interpolation='none',
                           cmap='rainbow')
            # im = ax.imshow(dose_3d[ax.index, :, :], alpha=0.4 * (dose_3d[ax.index, :, :] > 0), interpolation='none',
            #                cmap='rainbow')
            # import matplotlib.ticker as ticker
            plt.colorbar(im)

        if show_isodose:
            if not show_dose:
                dose_3d = self.create_3d_dose()
                self.dose_3d = dose_3d
            dose_legend = self.legend_dose_storage()
            self.dose_legend = dose_legend
            ax.contour(dose_3d[ax.index, :, :], dose_legend['dose value'],
                       colors=dose_legend['dose color'],
                       linewidths=0.5, zorder=2)
            dose_list = []
            for item in range(0, len(dose_legend['dose name'])):
                dose_list.append(Line2D([0], [0],
                                        color=dose_legend['dose color'][item],
                                        lw=1,
                                        label=dose_legend['dose name'][item]))
            ax.add_artist(ax.legend(handles=dose_list,
                                    bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.))
        ax.set_title('Axial View - Slice #: {}'.format(ax.index))

        # Create the Slider
        slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
        self.slider = Slider(ax=slider_ax, label="Slice", valmin=0, valmax=self.volume.shape[0] - 1, valinit=ax.index,
                             valstep=1)
        # self.slider.on_changed(self.update)

        # create check box for structures
        # rax = plt.axes()
        if show_struct:
            rax = plt.axes([0.05, 0.4, 0.2, 0.6])
            structs = self._structures.structures_dict['name']
            struct_masks = self._structures.structures_dict['structure_mask_3d']
            self.struct_masks = struct_masks
            all_mask = []
            # colors = Visualization.get_cmap(len(structs))
            colors = self.get_colors(30)
            # cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
            for i in range(len(structs)):
                cmap = mpl.colors.ListedColormap(colors[i])
                im = ax.imshow(struct_masks[i][ax.index, :, :], alpha=0.6 * (struct_masks[i][ax.index, :, :] > 0),
                               interpolation='none', cmap=cmap)
                all_mask.append(im)
            # self.all_mask = all_mask
            self.labels = [struct for struct in structs]
            visibility = [mask.get_visible() for mask in all_mask]
            self.check = CheckButtons(rax, self.labels, visibility)
            # [rec.set_facecolor(colors[i]) for i, rec in enumerate(check.rectangles)]
            # [label.set_color(colors[i]) for i, label in enumerate(check.labels)]
            [label.set_fontsize(10) for i, label in enumerate(self.check.labels)]
            # [rec.set_width(0.05) for i, rec in enumerate(check.rectangles)]
            for i, rec in enumerate(self.check.rectangles):
                rec.set_facecolor(colors[i])
                # rec.set_width(0.05)
                # rec.set_height(0.05)
                # x = rec.get_x()
                # y = rec.get_y()
                # check.lines[i][0].set_data([[x, x+0.05], [y+0.05, y]])
                # check.lines[i][1].set_data([[x, x + 0.05], [y, y+0.05]])
            # count = 0
            # for label in check.labels:
            #     label.set_facecolor(colors[count])
            #     count = count + 1
            # check = self.display_struct()
            # [rec.set_facecolor(c[i]) for i, rec in enumerate(check.rectangles)]
            self.check.on_clicked(self.func)
        self.slider.on_changed(self.update)
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        fig.canvas.mpl_connect('scroll_event', self.process_scroll)
        # plt.ioff()  # turns interactive mode OFF, should make show() blocking
        plt.show()
        # fig.show()

    # @staticmethod
    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'left':
            self.previous_slice(ax)
        elif event.key == 'right':
            self.next_slice(ax)
        fig.canvas.draw()

    def update(self, val):
        fig = self.fig
        ax = fig.axes[0]
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max)

        # Update the image's colormap
        st = time.time()
        ax.index = self.slider.val % self.volume.shape[0]
        ind_num = 0
        ax.images[ind_num].set_array(self.volume[ax.index, :, :])
        if self.show_dose:
            ind_num = ind_num + 1
            ax.images[ind_num].set_array(self.dose_3d[ax.index, :, :])
            ax.images[ind_num].set_alpha(0.4 * (self.dose_3d[ax.index, :, :] > 0))
        if self.show_isodose:
            # for c in ax.collections:
            #     # c.remove()
            for i in range(len(ax.collections) - 1, -1, -1):
                del ax.collections[i]

            # for i in range(len(ax.collections)):
            #     print(len(ax.collections))
            #     del ax.collections[i]
            # ax.collections = []
            ax.contour(self.dose_3d[ax.index, :, :], self.dose_legend['dose value'],
                       colors=self.dose_legend['dose color'],
                       linewidths=0.5, zorder=2)

        # if self.show_struct:
        #     for i in range(len(ax.images)):
        #         if self.show_dose:
        #             if i > 1:
        #                 ax.images[i].set_array(self.struct_masks[i-2][ax.index, :, :]) # i-2 since first two index are ct and dose
        #                 ax.images[i].set_alpha(0.6 * (self.struct_masks[i-2][ax.index, :, :] > 0))
        if self.show_struct:
            for i in range(len(self.struct_masks)):
                ind_num = ind_num + 1
                ax.images[ind_num].set_array(
                    self.struct_masks[i][ax.index, :, :])  # i-2 since first two index are ct and dose
                ax.images[ind_num].set_alpha(0.8 * (self.struct_masks[i][ax.index, :, :] > 0))
        ax.set_title('Axial View - Slice #: {}'.format(ax.index))
        fig.canvas.draw()
        end = time.time() - st
        print('time for changing slide {} s'.format(end))

    # @staticmethod
    def process_scroll(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.button == 'up':
            self.previous_slice(ax)
        elif event.button == 'down':
            self.next_slice(ax)
        fig.canvas.draw()

    # @staticmethod
    def previous_slice(self, ax):
        # volume = ax.volume
        # dose_3d = ax.dose_3d
        ax.index = (ax.index - 1) % self.volume.shape[0]  # wrap around using %
        # self.slider.val = ax.index
        ind_num = 0
        ax.images[ind_num].set_array(self.volume[ax.index, :, :])
        if self.show_dose:
            ind_num = ind_num + 1
            ax.images[ind_num].set_array(self.dose_3d[ax.index, :, :])
            ax.images[ind_num].set_alpha(0.4 * (self.dose_3d[ax.index, :, :] > 0))
        if self.show_isodose:
            for i in range(len(ax.collections) - 1, -1, -1):
                del ax.collections[i]
            ax.contour(self.dose_3d[ax.index, :, :], self.dose_legend['dose value'],
                       colors=self.dose_legend['dose color'],
                       linewidths=0.5, zorder=2)
        if self.show_struct:
            for i in range(len(self.struct_masks)):
                ind_num = ind_num + 1
                ax.images[ind_num].set_array(
                    self.struct_masks[i][ax.index, :, :])  # i-2 since first two index are ct and dose
                ax.images[ind_num].set_alpha(0.8 * (self.struct_masks[i][ax.index, :, :] > 0))

        ax.set_title('Axial View - Slice #: {}'.format(ax.index))
        # plt.title = 'Slice# {}'.format(ax.index)

    # @staticmethod
    def next_slice(self, ax):
        # volume = ax.volume
        # dose_3d = ax.dose_3d
        ax.index = (ax.index + 1) % self.volume.shape[0]
        ind_num = 0
        ax.images[ind_num].set_array(self.volume[ax.index, :, :])
        if self.show_dose:
            ind_num = ind_num + 1
            ax.images[ind_num].set_array(self.dose_3d[ax.index, :, :])
            ax.images[ind_num].set_alpha(0.4 * (self.dose_3d[ax.index, :, :] > 0))
        if self.show_isodose:
            for i in range(len(ax.collections) - 1, -1, -1):
                del ax.collections[i]
            ax.contour(self.dose_3d[ax.index, :, :], self.dose_legend['dose value'],
                       colors=self.dose_legend['dose color'],
                       linewidths=0.5, zorder=2)
        if self.show_struct:
            for i in range(len(self.struct_masks)):
                ind_num = ind_num + 1
                ax.images[ind_num].set_array(
                    self.struct_masks[i][ax.index, :, :])  # i-2 since first two index are ct and dose
                ax.images[ind_num].set_alpha(0.8 * (self.struct_masks[i][ax.index, :, :] > 0))
        ax.set_title('Axial View - Slice #: {}'.format(ax.index))
        # plt.title = 'Slice# {}'.format(ax.index)

    def create_3d_dose(self, normalize=True):
        dose_vox_map = self._structures.opt_voxels_dict['ct_to_dose_voxel_map'][0]
        dose_1d = self._structures.opt_voxels_dict['dose_1d']
        dose_3d = np.zeros_like(dose_vox_map, dtype=float)
        inds = np.unique(dose_vox_map[dose_vox_map >= 0])
        a = np.where(np.isin(dose_vox_map, inds))
        dose_3d[a] = dose_1d[dose_vox_map[a]]
        if normalize:
            dose_3d = dose_3d * self._clinical_criteria.clinical_criteria_dict['num_of_fractions']
        # for i in range(len(inds)):
        #     ind = inds[i]
        #     if dose_1d[ind] > 0:
        #         dose_2d[np.where(dose_vox_map_2d == ind)] = dose_1d[ind]
        return dose_3d
        # plt.imshow(dose_3d[x, :, :], alpha=0.6, cmap='rainbow')

    # Make checkbuttons with all plotted lines with correct visibility
    def display_struct(self):
        fig = self.fig
        ax = fig.axes[0]
        rax = plt.axes([0.05, 0.4, 0.1, 0.15])
        structs = self._structures.structures_dict['name']
        struct_mask = self._structures.structures_dict['structure_mask_3d']
        all_mask = []
        colors = self.get_colors(30)
        for i in range(len(structs)):
            im = ax.imshow(struct_mask[i][ax.index, :, :], alpha=0.6 * (struct_mask[i][ax.index, :, :] > 0),
                           cmap=colors[i])
            all_mask.append(im)
        self.all_mask = all_mask
        self.labels = [struct for struct in structs]
        visibility = [mask.get_visible() for mask in all_mask]
        check = CheckButtons(rax, self.labels, visibility)

        return check

    def func(self, label):
        fig = self.fig
        ax = fig.axes[0]
        st = time.time()
        index = self.labels.index(label)
        # self.all_mask[index].set_visible(not self.all_mask[index].get_visible())
        if self.show_dose:
            ax.images[index + 2].set_visible(not ax.images[index + 2].get_visible())
        else:
            ax.images[index + 1].set_visible(not ax.images[index + 1].get_visible())
        fig.canvas.draw()
        end = time.time() - st
        print('time for changing struct {} s'.format(end))

    @staticmethod
    def get_cmap_colors(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def legend_dose_storage(self):
        # dose_color = [[0.55, 0, 1], [0, 0, 1], [0, 0.5, 1], [0, 1, 0],
        #               [1, 1, 0], [1, 0.65, 0], [1, 0, 0], [0.55, 0, 0]]
        dose_color = [[0.55, 0, 0], [0, 0, 1], [0.55, 0, 1], [0, 0.5, 1], [0, 1, 0], [1, 0, 0]]
        dose_level = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1]
        dose_prescription = self._clinical_criteria.clinical_criteria_dict['pres_per_fraction_gy'] * \
                            self._clinical_criteria.clinical_criteria_dict[
                                'num_of_fractions']
        dose_value = [item * dose_prescription for item in dose_level]
        dose_name = []
        for item in range(0, len(dose_level)):
            dose_name.append(str(round(dose_level[item] * 100, 2)) + ' % / ' + \
                             str(round(dose_value[item], 3)) + ' ' + \
                             'Gy')
        dose_storage_legend = {'dose color': dose_color, 'dose level': dose_level, \
                               'dose value': dose_value, 'dose name': dose_name}
        return dose_storage_legend
