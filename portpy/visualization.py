import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import numpy as np
from portpy.evaluation import Evaluation
from matplotlib.lines import Line2D
import os
from portpy.load_metadata import load_metadata
import pandas as pd
import webbrowser

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


# mpl.use('TkAgg')  # or can use 'Qt5Agg', whatever you have/prefer


class Visualization:
    dose_type = Literal["Absolute(Gy)", "Relative(%)"]
    volume_type = Literal["Absolute(cc)", "Relative(%)"]

    @staticmethod
    def plot_robust_dvh(my_plan, dose_list, structs=None, style='solid', norm_flag=False, norm_volume=90,
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
            colors = my_plan.get_colors(30)
        if structs is None:
            # orgs = []
            structs = my_plan.structures.structures_dict['name']
        max_dose = 0.0
        all_orgs = my_plan.structures.structures_dict['name']
        # orgs = [org.upper for org in orgs]
        pres = my_plan.clinical_criteria['pres_per_fraction_gy'] * my_plan.clinical_criteria[
            'num_of_fractions']
        legend = []
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.size'] = 12
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in structs:
                continue
            dose_sort_list = []
            for dose in dose_list:
                x, y = my_plan.get_dvh(dose, all_orgs[i], weight_flag=weight_flag)
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
                        norm_factor = Evaluation.get_dose(dose_list[scene_num], my_plan, norm_struct,
                                                          norm_volume) / pres
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
            x = my_plan.clinical_criteria['pres_per_fraction_gy'] * my_plan._clinical_criteria[
                'num_of_fractions'] * np.ones_like(y)
        else:
            x = my_plan.clinical_criteria['pres_per_fraction_gy'] * np.ones_like(y)
        plt.plot(x, y, color='black')
        if title:
            plt.title(title)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)

    @staticmethod
    def plot_dvh(my_plan, sol, dose_1d=None, structs=None, options_norm=None, options_fig=None,
                 dose_scale: dose_type = "Absolute(Gy)",
                 volume_scale: volume_type = "Relative(%)"):
        """
        Create dvh plot for the selected structures
        :param dose_1d: dose_1d in 1d
        :param volume_scale: volume scale on y-axis.
        :param dose_scale: dose_1d scale on x axis. Default= absolute(Gy)
        :param sol: optimal sol dictionary
        :param my_plan: object of class Plan
        :param structs: structures to be included in dvh plot
        :param options_norm: normalization options. e.g. options_norm['norm_flag] = True will normalize the plan
        :param options_fig: figure options. e.g. options_fig['style'] = 'solid' will set the style of dvh plot
        :return: dvh plot for the selected structures

        """
        if options_fig is None:
            options_fig = {}
        if options_norm is None:
            options_norm = {}
            # dose_1d = self.beams.get_influence_matrix() @ self.beams.optimal_intensity
            # dose_1d = my_plan.structures.opt_voxels_dict['dose_1d']
        if dose_1d is None:
            if 'dose_1d' not in sol:
                sol['dose_1d'] = sol['inf_matrix'].A * sol['optimal_intensity'] * my_plan.get_num_of_fractions()
                dose_1d = sol['dose_1d']
            else:
                dose_1d = sol['dose_1d']
        # getting options_fig:
        style = options_fig['style'] if 'style' in options_fig else 'solid'
        width = options_fig['width'] if 'width' in options_fig else None
        colors = options_fig['colors'] if 'colors' in options_fig else None
        legend_font_size = options_fig['legend_font_size'] if 'legend_font_size' in options_fig else 10
        figsize = options_fig['figsize'] if 'figsize' in options_fig else (12, 8)
        title = options_fig['title'] if 'title' in options_fig else None
        filename = options_fig['filename'] if 'filename' in options_fig else None
        show = options_fig['show'] if 'show' in options_fig else True

        # getting norm options
        norm_flag = options_norm['norm_flag'] if 'norm_flag' in options_norm else False
        norm_volume = options_norm['norm_volume'] if 'norm_volume' in options_norm else 90
        norm_struct = options_norm['norm_struct'] if 'norm_struct' in options_norm else 'PTV'

        plt.rcParams['font.size'] = 12
        if width is None:
            if style == 'dotted' or style == 'dashed':
                width = 2.5
            else:
                width = 2
        if colors is None:
            colors = Visualization.get_colors(30)
        if structs is None:
            # orgs = []
            structs = my_plan.structures.structures_dict['name']
        max_dose = 0.0
        max_vol = 0.0
        all_orgs = my_plan.structures.structures_dict['name']
        # orgs = [org.upper for org in orgs]
        pres = my_plan.get_prescription()
        legend = []
        fig = plt.figure(figsize=figsize)
        if norm_flag:
            norm_factor = Evaluation.get_dose(sol, struct=norm_struct, volume_per=norm_volume) / pres
            dose_1d = dose_1d / norm_factor
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in structs:
                continue
            # for dose_1d in dose_list:
            #
            x, y = Evaluation.get_dvh(sol, dose_1d=dose_1d, struct=all_orgs[i])
            if dose_scale == 'Absolute(Gy)':
                max_dose = np.maximum(max_dose, x[-1])
                plt.xlabel('Dose (Gy)')
            elif dose_scale == 'Relative(%)':
                x = x / pres * 100
                max_dose = np.maximum(max_dose, x[-1])
                plt.xlabel('Dose (%)')

            if volume_scale == 'Absolute(cc)':
                y = y * my_plan.structures.get_volume_cc(all_orgs[i]) / 100
                max_vol = np.maximum(max_vol, y[1] * 100)
                plt.ylabel('Volume (cc)')
            elif volume_scale == 'Relative(%)':
                max_vol = np.maximum(max_vol, y[0] * 100)
                plt.ylabel('Volume Fraction (%)')
            plt.plot(x, 100 * y, linestyle=style, linewidth=width, color=colors[i])
            legend.append(all_orgs[i])

        # plt.xlabel('Dose (Gy)')
        # plt.ylabel('Volume Fraction (%)')
        plt.xlim(0, max_dose * 1.1)
        plt.ylim(0, max_vol)
        plt.legend(legend, prop={'size': legend_font_size}, loc="upper right")
        plt.grid(b=True, which='major', color='#666666', linestyle='-')

        # Show the minor grid lines with very faint and almost transparent grey lines
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        y = np.arange(0, 101)
        # if norm_flag:
        #     x = pres * np.ones_like(y)
        # else:
        if dose_scale == "Absolute(Gy)":
            x = pres * np.ones_like(y)
        else:
            x = 100 * np.ones_like(y)

        plt.plot(x, y, color='black')
        if title:
            plt.title(title)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)

    @staticmethod
    def plot_binary_mask_points(my_plan, structure, show=True, color=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ind = my_plan.structures.structures_dict['name'].index(structure)
        mask_3d = my_plan.structures.structures_dict['structure_mask_3d'][ind]
        pos = np.where(mask_3d == 1)
        if color is None:
            color = Visualization.get_colors(0)
        ax.scatter(pos[0], pos[1], pos[2], c=color)
        if show:
            plt.show()

    @staticmethod
    def get_colors(num):
        # random.seed(42)
        # colors = []
        # for i in range(num):
        #     color = (random.random(), random.random(), random.random())
        #     colors.append(color)
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
                  '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
                  '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                  '#000075', '#808080', '#ffffff', '#000000']
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

    @staticmethod
    def plot_fluence_2d(my_plan, beam_id: int, optimal_fluence_2d=None):
        """

        :param optimal_fluence_2d:
        :param my_plan: object of class Plan
        :param beam_id: beam_id of the beam
        :return: 2d optimal fluence plot
        """
        my_plan.inf_matrix.plot_fluence_2d(beam_id=beam_id, optimal_fluence_2d=optimal_fluence_2d)

    @staticmethod
    def plot_fluence_3d(my_plan, beam_id: int, optimal_fluence_2d=None):
        """

                :param optimal_fluence_2d:
                :param my_plan: object of class Plan
                :param beam_id: beam_id of the beam
                :return: 3d optimal fluence plot
                """
        my_plan.inf_matrix.plot_fluence_3d(beam_id=beam_id, optimal_fluence_2d=optimal_fluence_2d)

    @staticmethod
    def plot_2d_dose(my_plan, sol, slice_num=40, structs=None, show_dose=True, show_struct=True, show_isodose=True,
                     dpi=100):
        """
        :param sol: solution to optimization
        :param my_plan: object of class Plan
        :param slice_num: slice number
        :param structs: structures
        :param show_dose: view dose_1d on the slice
        :param show_struct: view structure on the slice
        :param show_isodose: view isodose
        :param dpi: Default dpi=100 for figure
        :return: plot 2d view of ct, dose_1d, isodose and structures
        """
        fig, ax = plt.subplots(dpi=dpi)
        plt.rcParams["figure.autolayout"] = True
        ct = my_plan.ct['ct_hu_3d'][0]
        # adjust the main plot to make room for the legends
        fig.subplots_adjust(left=0.2)
        dose_3d = []
        if show_dose:
            dose_3d = sol['inf_matrix'].dose_1d_to_3d(sol=sol)
            ax.imshow(ct[slice_num, :, :], cmap='gray')
            # im = ax.imshow(dose_3d[slice_num, :, :], alpha=0.4 * (dose_3d[slice_num, :, :] > 0), interpolation='none',
            #                cmap='rainbow')
            masked = np.ma.masked_where(dose_3d[slice_num, :, :] <= 0, dose_3d[slice_num, :, :])
            im = ax.imshow(masked, alpha=0.4, interpolation='none',
                           cmap='rainbow')

            plt.colorbar(im)

        if show_isodose:
            if not show_dose:
                dose_3d = my_plan.inf_matrix.dose_1d_to_3d(sol=sol)
            dose_legend = Visualization.legend_dose_storage(my_plan)
            ax.contour(dose_3d[slice_num, :, :], dose_legend['dose_1d value'],
                       colors=dose_legend['dose_1d color'],
                       linewidths=0.5, zorder=2)
            dose_list = []
            for item in range(0, len(dose_legend['dose_1d name'])):
                dose_list.append(Line2D([0], [0],
                                        color=dose_legend['dose_1d color'][item],
                                        lw=1,
                                        label=dose_legend['dose_1d name'][item]))
            ax.add_artist(ax.legend(handles=dose_list,
                                    bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0.))
        ax.set_title('Axial View - Slice #: {}'.format(slice_num))

        # create check box for structures
        # rax = plt.axes()
        if show_struct:
            # rax = plt.axes([0.05, 0.4, 0.2, 0.6])
            if structs is None:
                structs = my_plan.structures.structures_dict['name']
            struct_masks = my_plan.structures.structures_dict['structure_mask_3d']
            all_mask = []
            colors = Visualization.get_colors(30)
            for i in range(len(structs)):
                ind = my_plan.structures.structures_dict['name'].index(structs[i])
                cmap = mpl.colors.ListedColormap(colors[i])
                # im = ax.imshow(struct_masks[ind][slice_num, :, :], alpha=0.6 * (struct_masks[ind][slice_num, :, :] > 0),
                #                interpolation='none', cmap=cmap)
                masked = np.ma.masked_where(struct_masks[ind][slice_num, :, :] == 0, struct_masks[ind][slice_num, :, :])
                im = ax.imshow(masked, alpha=0.6,
                               interpolation='none', cmap=cmap)
            labels = [struct for struct in structs]
            # get the colors of the values, according to the
            # colormap used by imshow
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # rax.labels = labels
            fig.legend(handles=patches, bbox_to_anchor=(0.1, 0.8), loc=2, borderaxespad=0.,
                       bbox_transform=fig.transFigure)
        plt.show()
        # fig.show()

    @staticmethod
    def get_cmap_colors(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    @staticmethod
    def legend_dose_storage(my_plan):
        # dose_color = [[0.55, 0, 1], [0, 0, 1], [0, 0.5, 1], [0, 1, 0],
        #               [1, 1, 0], [1, 0.65, 0], [1, 0, 0], [0.55, 0, 0]]
        dose_color = [[0.55, 0, 0], [0, 0, 1], [0.55, 0, 1], [0, 0.5, 1], [0, 1, 0], [1, 0, 0]]
        dose_level = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1]
        dose_prescription = my_plan.clinical_criteria.clinical_criteria_dict['pres_per_fraction_gy'] * \
                            my_plan.clinical_criteria.clinical_criteria_dict['num_of_fractions']
        dose_value = [item * dose_prescription for item in dose_level]
        dose_name = []
        for item in range(0, len(dose_level)):
            dose_name.append(str(round(dose_level[item] * 100, 2)) + ' % / ' +
                             str(round(dose_value[item], 3)) + ' ' + 'Gy')
        dose_storage_legend = {'dose_1d color': dose_color, 'dose_1d level': dose_level, 'dose_1d value': dose_value,
                               'dose_1d name': dose_name}
        return dose_storage_legend

    @staticmethod
    def view_in_slicer(my_plan, slicer_path=None, img_dir=None):
        """
        :param img_dir: directory where nrrd images are saved.
        :param my_plan: object of class Plan
        :param slicer_path: slicer executable path on your local machine
        :return: plot the images in 3d slicer

        view ct, dose_1d and structure set images in 3d slicer
        """
        if slicer_path is None:
            slicer_path = r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe'
        if img_dir is None:
            img_dir = os.path.join(os.getcwd(), "..", 'Data', my_plan.patient_id)
        slicer_script_dir = os.path.join(os.getcwd(), 'portpy', 'slicer_script.py')
        # patient_folder_path = os.path.join(os.getcwd(), "..", 'Data', my_plan.patient_id)
        # import SimpleITK as sitk

        # ct_arr = self._ct['ct_hu_3D'][0]
        # ct = sitk.GetImageFromArray(ct_arr)
        # ct.SetOrigin(self._ct['origin_xyz_mm'])
        # ct.SetSpacing(self._ct['resolution_xyz_mm'])
        # ct.SetDirection(self._ct['direction'])
        # sitk.WriteImage(ct, os.path.join(patient_folder_path, 'ct.nrrd'))
        #
        # dose_arr = self.create_3d_dose()
        # dose_1d = sitk.GetImageFromArray(dose_arr)
        # dose_1d.SetOrigin(self._ct['origin_xyz_mm'])
        # dose_1d.SetSpacing(self._ct['resolution_xyz_mm'])
        # dose_1d.SetDirection(self._ct['direction'])
        # sitk.WriteImage(dose_1d, os.path.join(patient_folder_path, 'dose_1d.nrrd'))
        #
        # labels = self._structures.structures_dict['structure_mask_3d']
        # mask_arr = np.array(labels).transpose((1, 2, 3, 0))
        # mask = sitk.GetImageFromArray(mask_arr)
        # mask.SetOrigin(self._ct['origin_xyz_mm'])
        # mask.SetSpacing(self._ct['resolution_xyz_mm'])
        # mask.SetDirection(self._ct['direction'])
        # sitk.WriteImage(mask, os.path.join(patient_folder_path, 'rtss.seg.nrrd'), True)

        # mask_3d = self._structures.structures_dict['structure_mask_3d']
        # structs = self._structures.structures_dict['name']
        # filename = os.path.join(patient_folder_path, 'tmp_data')
        # np.savez(filename, ct=ct_arr, dose_1d=dose_arr, mask_3d=mask_3d)
        # tmp_dict = {}
        # tmp_dict['structs'] = structs
        # tmp_dict['origin_xyz_mm'] = self._ct['origin_xyz_mm']
        # # tmp_dict['resolution_xyz_mm'] = self._ct['resolution_xyz_mm']
        # import json
        # filename = os.path.join(patient_folder_path, 'tmp_dict.json')
        # with open(filename, 'w') as fp:
        #     json.dump(tmp_dict, fp)
        # dose_1d = sitk.GetImageFromArray(dose_arr)
        # dose_1d.SetOrigin(self._ct['origin_xyz_mm'])
        # dose_1d.SetSpacing(self._ct['resolution_xyz_mm'])
        # dose_1d.SetDirection(self._ct['direction'])
        # sitk.WriteImage(dose_1d, os.path.join(patient_folder_path, 'dose_1d.nrrd'))
        # plan_file = r'//pisidsmph/Treatplanapp/ECHO/Research/Data_newformat/Python-PORT/my_plan'
        # slicer_script_file = r'\\pisidsmph\Treatplanapp\ECHO\Research\Data_newformat\Python-PORT\portpy\slicer_script.py'
        import subprocess
        # subprocess.run([slicer_path, ' --python-script', f' {slicer_script_file}'])
        # subprocess.run(
        #     [slicer_path, '--python-code',
        #      'slicer.util.loadVolume', '(',
        #      os.path.join(patient_folder_path, "CT.nrrd"), ')', ';',
        #      'slicer.util.setSliceViewerLayers(background=CT);',
        #      'slicer.util.loadVolume', '(',
        #      os.path.join(patient_folder_path, "dose_1d.nrrd"), ')', ';',
        #      'slicer.util.setSliceViewerLayers(foreground=dose_1d)', ';',
        #      'slicer.util.setSliceViewerLayers(foregroundOpacity=0.4)',
        #      'for color in [''Red'', ''Yellow'', ''Green'']:',
        #      '  slicer.app.layoutManager().sliceWidget(color).sliceLogic().GetSliceCompositeNode().SetForegroundVolumeID(dose_1d.GetID());',
        #      '  slicer.app.layoutManager().sliceWidget(color).sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(scanList[-1].GetID());'
        #      ], shell=False)
        # subprocess.run([slicer_path, f'--python-script "{os.getcwd()}\portpy\slicer_script.py" --no-splash'], shell=False)
        # subprocess.run(
        #     r'{} --python-script .\portpy\slicer_script.py {}'.format(slicer_path, self.patient_name), shell=False)
        # subprocess.run([slicer_path, f"--python-script", slicer_script_dir, img_dir,
        #                 [name + ',' for name in my_plan.structures.structures_dict['name']]], shell=False)
        subprocess.run([slicer_path, f"--python-script", slicer_script_dir, img_dir,
                        ','.join(my_plan.structures.structures_dict['name'])], shell=False)
        #         subprocess.run([slicer_path, '--python-code',
        #                         f"""slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'ct');
        # ct_node = slicer.util.getNode('ct');
        # ct_node.SetOrigin({self._ct['origin_xyz_mm']});
        # ct_node.SetSpacing({self._ct['resolution_xyz_mm']});
        # slicer.util.updateVolumeFromArray(ct_node, {ct_arr});
        # slicer.util.setSliceViewerLayers(foreground=ct_node);
        # slicer.util.setSliceViewerLayers(foregroundOpacity=0.4);"""], shell=False)
        print('Done')
        # doseNode.SetIJKToRASDirections(ijkMat)
        # 'seg', '=', 'slicer.util.loadSegmentation', '(','/home/oguzcan-bekar/Desktop/PyQt/mask.nii.gz', ');', 'seg.CreateClosedSurfaceRepresentation();'], shell=False)

    @staticmethod
    def display_patient_metadata(pat_name, pat_dir=None, show_beams=True, show_structs=True):
        if pat_dir is None:
            pat_dir = os.path.join(os.getcwd(), "..", 'Data')
            pat_dir = os.path.join(pat_dir, pat_name)
        else:
            pat_dir = os.path.join(pat_dir, pat_name)
        options = {'loadInfluenceMatrixFull': 1}
        meta_data = load_metadata(pat_dir, options=options)
        if show_beams:
            beams = meta_data['beams']
            del beams['beamlets']
            df = pd.DataFrame.from_dict(beams)
            is_full = df['influenceMatrixFull_File'].str.contains('full', na=False)
            is_sparse = df['influenceMatrixSparse_File'].str.contains('sparse', na=False)
            for ind, (sparse, full) in enumerate(zip(is_full, is_sparse)):
                if sparse and full:
                    df.at[ind, 'sparse/full'] = 'sparse/full'
                elif sparse and not full:
                    df.at[ind, 'sparse/full'] = 'sparse'
                elif not sparse and full:
                    df.at[ind, 'sparse/full'] = 'full'

            keep_columns = ['ID', 'gantry_angle', 'collimator_angle', 'couch_angle', 'beam_modality', 'energy_MV',
                            'sparse/full',
                            'iso_center', 'MLC_name',
                            'machine_name']
            df = df[keep_columns]
            # # print('Beams table..')
            # print(tabulate(df, headers='keys', tablefmt='psql'))
        if show_structs:
            print('\n\nStructures table..')
            structures = meta_data['structures']
            struct_df = pd.DataFrame.from_dict(structures)
            keep_columns = ['name', 'volume_cc']
            struct_df = struct_df[keep_columns]
            # print(tabulate(df, headers='keys', tablefmt='psql'))
        # OUTPUT AN HTML FILE
        html_string = '''
                <html>
                  <head><title>Portpy MetaData</title></head>
                  <link rel="stylesheet" type="text/css" href="df_style.css"/>
                  <body>
                  <h1> PortPy Metadata </h1> 
                  <h4> Beams Metadata </h4>
                    {table_1}
                  <h4> Structures Metadata </h4>
                    {table_2}
                  </body>
                </html>.
                '''
        with open('temp.html', 'w') as f:
            f.write(html_string.format(table_1=df.to_html(index=False, header=True, classes='mystyle'),
                                       table_2=struct_df.to_html(index=False, header=True, classes='mystyle')))
        webbrowser.open('file://' + os.path.realpath('temp.html'))

    @staticmethod
    def display_patients(data_dir=None):
        display_dict = {}
        if data_dir is None:
            data_dir = os.path.join(os.getcwd(), "..", 'Data')
            # pat_dir = os.path.join(pat_dir, pat_name)
        pat_names = os.listdir(data_dir)
        for i, pat_name in enumerate(pat_names):
            if pat_name == 'ECHO_PROST_1' or pat_name == 'Lung_Patient_1':
                display_dict.setdefault('patient_name', []).append(pat_name)
                meta_data = load_metadata(os.path.join(data_dir, pat_name), options=None)
                display_dict.setdefault('disease_site', []).append(meta_data['clinical_criteria']['disease_site'])
                ind = meta_data['structures']['name'].index('PTV')
                display_dict.setdefault('ptv_vol_cc', []).append(meta_data['structures']['volume_cc'][ind])
                display_dict.setdefault('num_beams', []).append(len(meta_data['beams']['ID']))
                res = all(ele == meta_data['beams']['iso_center'][0] for ele in meta_data['beams']['iso_center'])
                if res:
                    display_dict.setdefault('iso_center_shift ', []).append('No')
                else:
                    display_dict.setdefault('iso_center_shift ', []).append('Yes')
        df = pd.DataFrame.from_dict(display_dict)
        html_string = '''
                <html>
                  <head><title>Portpy MetaData</title></head>
                  <link rel="stylesheet" type="text/css" href="df_style.css"/>
                  <body>
                  <h4> Patients Metadata </h4>
                    {table}
                  </body>
                </html>.
                '''
        with open('temp.html', 'w') as f:
            f.write(html_string.format(table=df.to_html(index=False, header=True, classes='mystyle')))
        webbrowser.open('file://' + os.path.realpath('temp.html'))
        # print(tabulate(df, headers='keys', tablefmt='psql'))

    @staticmethod
    def plan_metrics(my_plan, sol):
        # convert clinical criteria in dataframe
        display_dict = {}
        df = pd.DataFrame.from_dict(my_plan.clinical_criteria.clinical_criteria_dict['criteria'])
        for ind in range(len(df)):
            if df.name[ind] == 'max_dose':
                struct = df.parameters[ind]['structure_name']
                max_dose = Evaluation.get_max_dose(sol, structure_name=struct)
                if 'limit_dose_gy' in df.constraints[ind] or 'goal_dose_gy' in df.constraints[ind]:
                    df.at[ind, 'Plan Value'] = max_dose
                elif 'limit_dose_perc' in df.constraints[ind] or 'goal_dose_perc' in df.constraints[ind]:
                    df.at[ind, 'Plan Value'] = max_dose / (
                                my_plan.get_prescription() * my_plan.get_num_of_fractions()) * 100
            if df.name[ind] == 'mean_dose':
                struct = df.parameters[ind]['structure_name']
                mean_dose = Evaluation.get_mean_dose(sol, structure_name=struct)
                df.at[ind, 'Plan Value'] = mean_dose
            if df.name[ind] == 'dose_volume_V':
                struct = df.parameters[ind]['structure_name']
                if 'limit_volume_perc' in df.constraints[ind] or 'goal_volume_perc' in df.constraints[ind]:
                    dose = df.parameters[ind]['dose_gy']
                    volume = Evaluation.get_volume(sol, struct=struct, dose_value=dose)
                    df.at[ind, 'Plan Value'] = volume
                elif 'limit_volume_cc' in df.constraints[ind] or 'goal_volume_cc' in df.constraints[ind]:
                    dose = df.parameters[ind]['dose_gy']
                    volume = Evaluation.get_volume(sol, struct=struct, dose_value=dose)
                    vol_cc = my_plan.structures.get_volume_cc(structure_name=struct)*volume/100
                    df.at[ind, 'Plan Value'] = vol_cc

        def color_plan_value(row):

            highlight = 'background-color: red;'
            highlight_green = 'background-color: green;'
            highlight_brown = 'background-color: yellow;'
            default = ''

            def matching_keys(dictionary, search_string):
                get_key = None
                for key, val in dictionary.items():
                    if search_string in key:
                        get_key = key
                if get_key is not None:
                    return get_key
                else:
                    return ''

            limit_key = matching_keys(row['constraints'], 'limit')
            goal_key = matching_keys(row['constraints'], 'goal')

            row_color = [default, default]
            # must return one string per cell in this row
            if limit_key in row['constraints']:
                if row['Plan Value'] > row['constraints'][limit_key]:
                    row_color = [highlight, default]
                else:
                    row_color = [highlight_green, default]
            if goal_key in row['constraints']:
                if row['Plan Value'] > row['constraints'][goal_key]:
                    row_color = [highlight_brown, default]
                else:
                    row_color = [highlight_green, default]
            return row_color

        styled_df = df.style.apply(color_plan_value, subset=['Plan Value', 'constraints'], axis=1)
        html = styled_df.render()
        html_string = '''
                        <html>
                          <head><title>Portpy Clinical Criteria Evaluation</title></head>
                          <link rel="stylesheet" type="text/css" href="df_style.css"/>
                          <body>
                          <h4> Clinical Criteria</h4>
                            {table}
                          </body>
                        </html>.
                        '''
        with open('temp.html', 'w') as f:
            f.write(html_string.format(table=html))

        # with open('cc.html', 'w') as f:
        #     f.write(html)
        webbrowser.open('file://' + os.path.realpath('temp.html'))
