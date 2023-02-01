from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage import measure
from tabulate import tabulate
from portpy.evaluation import Evaluation
from matplotlib.lines import Line2D
import os
from portpy.utils import load_metadata
import pandas as pd
import webbrowser
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from portpy.plan import Plan
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
            x = my_plan.clinical_criteria['pres_per_fraction_gy'] * my_plan.clinical_criteria[
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
    def plot_dvh(my_plan: Plan, sol: dict = None, dose_1d: np.ndarray = None, structs: List[str] = None,
                 dose_scale: dose_type = "Absolute(Gy)",
                 volume_scale: volume_type = "Relative(%)", **options):
        """
        Create dvh plot for the selected structures

        :param my_plan: object of class Plan
        :param sol: optimal sol dictionary
        :param dose_1d: dose_1d in 1d voxels
        :param structs: structures to be included in dvh plot
        :param volume_scale: volume scale on y-axis. Default= Absolute(cc). e.g. volume_scale = "Absolute(cc)" or volume_scale = "Relative(%)"
        :param dose_scale: dose_1d scale on x axis. Default= Absolute(Gy). e.g. dose_scale = "Absolute(Gy)" or dose_scale = "Relative(%)"
        :keyword style (str): line style for dvh curve. default "solid". can be "dotted", "dash-dotted".
        :keyword width (int): width of line. Default 2
        :keyword colors(list): list of colors
        :keyword legend_font_size: Set legend_font_size. default 10
        :keyword figsize: Set figure size for the plot. Default figure size (12,8)
        :keyword create_fig: Create a new figure. Default True. If False, append to the previous figure
        :keyword title: Title for the figure
        :keyword filename: Name of the file to save the figure in current directory
        :keyword show: Show the figure. Default is True. If false, next plot can be append to it
        :keyword norm_flag: Use to normalize the plan. Default is False.
        :keyword norm_volume: Use to set normalization volume. default is 90 percentile.
        :return: dvh plot for the selected structures

        :Example:
        >>> Visualization.plot_dvh(my_plan, sol=sol, structs=['PTV', 'ESOPHAGUS'], dose_scale='Absolute(Gy)',volume_scale="Relative(%)", show=False, create_fig=True )
        """

        if dose_1d is None:
            if 'dose_1d' not in sol:
                dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'] * my_plan.get_num_of_fractions())
            else:
                dose_1d = sol['dose_1d']

        if sol is None:
            sol = dict()
            sol['inf_matrix'] = my_plan.inf_matrix  # create temporary solution

        # getting options_fig:
        style = options['style'] if 'style' in options else 'solid'
        width = options['width'] if 'width' in options else None
        colors = options['colors'] if 'colors' in options else None
        legend_font_size = options['legend_font_size'] if 'legend_font_size' in options else 10
        figsize = options['figsize'] if 'figsize' in options else (12, 8)
        title = options['title'] if 'title' in options else None
        filename = options['filename'] if 'filename' in options else None
        show = options['show'] if 'show' in options else True
        create_fig = options['create_fig'] if 'create_fig' in options else True
        show_criteria = options['show_criteria'] if 'show_criteria' in options else None

        # getting norm options
        norm_flag = options['norm_flag'] if 'norm_flag' in options else False
        norm_volume = options['norm_volume'] if 'norm_volume' in options else 90
        norm_struct = options['norm_struct'] if 'norm_struct' in options else 'PTV'

        plt.rcParams['font.size'] = 12
        if width is None:
            if style == 'dotted' or style == 'dashed':
                width = 2.5
            else:
                width = 2
        if colors is None:
            colors = Visualization.get_colors()
        if structs is None:
            # orgs = []
            structs = my_plan.structures.structures_dict['name']
        max_dose = 0.0
        max_vol = 0.0
        all_orgs = my_plan.structures.structures_dict['name']
        # orgs = [org.upper for org in orgs]
        pres = my_plan.get_prescription()
        legend = []
        if create_fig:
            plt.figure(figsize=figsize)
        if norm_flag:
            norm_factor = Evaluation.get_dose(sol, dose_1d=dose_1d, struct=norm_struct, volume_per=norm_volume) / pres
            dose_1d = dose_1d / norm_factor
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in structs:
                continue
            # for dose_1d in dose_list:
            #
            x, y = Evaluation.get_dvh(sol, struct=all_orgs[i], dose_1d=dose_1d)
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

        if show_criteria is not None:
            for s in range(len(show_criteria)):
                if 'dose_volume' in show_criteria[s]['name']:
                    x = show_criteria[s]['parameters']['dose_gy']
                    y = show_criteria[s]['constraints']['limit_volume_perc']
                    plt.plot(x, y, marker='x', color='red', markersize=20)
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
            plt.savefig(filename, bbox_inches="tight", dpi=300)

    @staticmethod
    def plot_binary_mask_points(my_plan, structure: str, show: bool = True, color: List[str] = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ind = my_plan.structures.structures_dict['name'].index(structure)
        mask_3d = my_plan.structures.structures_dict['structure_mask_3d'][ind]
        pos = np.where(mask_3d == 1)
        if color is None:
            color = Visualization.get_colors()
        ax.scatter(pos[0], pos[1], pos[2], c=color)
        if show:
            plt.show()

    @staticmethod
    def get_colors():
        """

        :return: return list of 20 colors
        """
        colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
                  '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
                  '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                  '#000075', '#808080', '#ffffff', '#000000']
        return colors

    @staticmethod
    def surface_plot(matrix: np.ndarray, **kwargs) -> None:
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, np.transpose(matrix), **kwargs)
        return fig, ax, surf

    @staticmethod
    def plot_fluence_2d(beam_id: int, sol: dict) -> None:
        """

        Displays fluence in 2d for the given beam_id

        :param beam_id: beam_id of the beam
        :param sol: solution dictionary after optimization
        :return: 2d optimal fluence plot

        :Example:
        >>> Visualization.plot_fluence_2d(beam_id=0, sol=sol)
        """
        sol['inf_matrix'].plot_fluence_2d(beam_id=beam_id, sol=sol)

    @staticmethod
    def plot_fluence_3d(beam_id: int, sol: dict) -> None:
        """
        Displays fluence in 3d for the given beam_id

        :param sol: solution after optimization
        :param beam_id: beam_id of the beam
        :return: 3d optimal fluence plot

        :Example:
        >>> Visualization.plot_fluence_3d(beam_id=0, sol=sol)
        """
        sol['inf_matrix'].plot_fluence_3d(beam_id=beam_id, sol=sol)

    @staticmethod
    def plot_2d_dose(my_plan: Plan, sol: dict, slice_num: int = 40, structs: List[str] = None, show_dose: bool = True,
                     show_struct: bool = True, show_isodose: bool = True,
                     dpi: int = 100) -> None:
        """

        Plot 2d view of ct, dose_1d, isodose and structure contours

        :param sol: solution to optimization
        :param my_plan: object of class Plan
        :param slice_num: slice number
        :param structs: structures for which contours to be displayed on the slice view. e.g. structs = ['PTV, ESOPHAGUS']
        :param show_dose: view dose_1d on the slice
        :param show_struct: view structure on the slice
        :param show_isodose: view isodose
        :param dpi: Default dpi=100 for figure
        :return: plot 2d view of ct, dose_1d, isodose and structure contours

        :Example:
        >>> Visualization.plot_2d_dose(my_plan, sol=sol, slice_num=50, structs=['PTV'], show_isodose=False)
        """
        fig, ax = plt.subplots(dpi=dpi)
        plt.rcParams["figure.autolayout"] = True
        ct = my_plan.ct['ct_hu_3d'][0]

        # adjust the main plot to make room for the legends
        fig.subplots_adjust(left=0.2)
        dose_3d = []
        if show_dose:
            dose_1d = sol['inf_matrix'].A * sol['optimal_intensity']*my_plan.get_num_of_fractions()
            dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
            ax.imshow(ct[slice_num, :, :], cmap='gray')
            masked = np.ma.masked_where(dose_3d[slice_num, :, :] <= 0, dose_3d[slice_num, :, :])
            im = ax.imshow(masked, alpha=0.4, interpolation='none',
                           cmap='rainbow')

            plt.colorbar(im)

        if show_isodose:
            if not show_dose:
                dose_1d = sol['inf_matrix'].A * sol['optimal_intensity'] * my_plan.get_num_of_fractions()
                dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
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
                                    bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.))
        ax.set_title('Axial View - Slice #: {}'.format(slice_num))

        if show_struct:
            if structs is None:
                structs = my_plan.structures.structures_dict['name']
            struct_masks = my_plan.structures.structures_dict['structure_mask_3d']
            all_mask = []
            colors = Visualization.get_colors()
            for i in range(len(structs)):
                ind = my_plan.structures.structures_dict['name'].index(structs[i])
                cmap = mpl.colors.ListedColormap(colors[i])
                contours = measure.find_contours(struct_masks[ind][slice_num, :, :], 0.5)
                for contour in contours:
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=colors[i])
            labels = [struct for struct in structs]
            # get the colors of the values, according to the
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # rax.labels = labels
            fig.legend(handles=patches, bbox_to_anchor=(0.1, 0.8), loc=2, borderaxespad=0.,
                       bbox_transform=fig.transFigure)
        plt.show()

    @staticmethod
    def get_cmap_colors(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    @staticmethod
    def legend_dose_storage(my_plan: Plan) -> dict:
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
    def view_in_slicer(my_plan: Plan, slicer_path: str = None, data_dir: str = None) -> None:
        """

        :param my_plan: object of class Plan
        :param slicer_path: slicer executable path on your local machine
        :param data_dir: the folder path where data located, defaults to None.
                If path = None, then it assumes the data is in sub-folder named Data in the current directory
        :return: plot the images in 3d slicer

        view ct, dose_1d and structure set images in 3d slicer

        :Example:
        >>> Visualization.view_in_slicer(my_plan, slicer_path='C:/Slicer/Slicer.exe', data_dir='path/to/nrrd/image')
        """
        if slicer_path is None:
            slicer_path = r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe'
        if data_dir is None:
            data_dir = os.path.join(Path(__file__).parents[1], 'Data', my_plan.patient_id)
        if not os.path.exists(data_dir):  # check if valid directory
            raise Exception("Invalid data directory. Please input valid directory")
        slicer_script_dir = os.path.join(Path(__file__).parents[0], 'utils', 'slicer_script.py')
        import subprocess
        subprocess.run([slicer_path, f"--python-script", slicer_script_dir, data_dir,
                        ','.join(my_plan.structures.structures_dict['name'])], shell=False)
        print('Done')

    @staticmethod
    def display_patient_metadata(patient_id: str, data_dir: str = None,
                                 in_browser: bool = False) -> None:
        """Displays the patient information in console or html format. If in_browswer is enabled
        it creates a temporary html file and lnunches your browser

        :param patient_id: the patient id
        :param data_dir: the folder path where data located, defaults to None.
                If path = None, then it assumes the data is in sub-folder named Data in the current directory
        :param in_browser: visualize in pretty way in browser. default to False. If false, plot table in console
        :raises invalid directory error: raises an exception if invalid data directory

        :Example:
        >>> Visualization.display_patient_metadata(patient_id=patient_id, data_dir='path/to/data', in_browser=True)
        """

        if data_dir is None:
            data_dir = os.path.join(Path(__file__).parents[1], 'Data')
            data_dir = os.path.join(data_dir, patient_id)
        else:
            data_dir = os.path.join(data_dir, patient_id)
        if not os.path.exists(data_dir):  # check if valid directory
            raise Exception("Invalid data directory. Please input valid directory")

        meta_data = load_metadata(data_dir)
        # if show_beams:
        # check if full and/or sparse influence matrices are provided.
        # Sparse matrix is just a truncated version of the full matrix (zeroing out small elements)
        # often used in the optimization for computational efficiency
        beams = meta_data['beams_dict']  # get beams_dict metadata
        beams_df = pd.DataFrame.from_dict(beams)  # using Pandas data structure
        is_full = beams_df['influenceMatrixFull_File'].str.contains('full',
                                                                    na=False)  # does the data include the full influence matrix
        is_sparse = beams_df['influenceMatrixSparse_File'].str.contains('sparse',
                                                                        na=False)  # does the data include the sparse influence matrix
        for ind, (sparse, full) in enumerate(zip(is_full, is_sparse)):  # create column for sparse/full check
            if sparse and full:
                beams_df.at[ind, 'influence_matrix(sparse/full)'] = 'Both'
            elif sparse and not full:
                beams_df.at[ind, 'influence_matrix(sparse/full)'] = 'Only Sparse'
            elif not sparse and full:
                beams_df.at[ind, 'influence_matrix(sparse/full)'] = 'Only Full'
        #  pick information to include in the table
        keep_columns = ['ID', 'gantry_angle', 'collimator_angle', 'couch_angle', 'beam_modality', 'energy_MV',
                        'influence_matrix(sparse/full)',
                        'iso_center', 'MLC_name',
                        'machine_name']
        beams_df = beams_df[keep_columns]

        structures = meta_data['structures']
        struct_df = pd.DataFrame.from_dict(structures)
        keep_columns = ['name', 'volume_cc']  # discard the columns except this columns
        struct_df = struct_df[keep_columns]

        # Write the results in a temporary html file in the current directory and launch a browser to display
        if in_browser:
            style_file = os.path.join(Path(__file__).parents[1], 'df_style.css')
            html_string = '''
                    <html>
                      <head><title>Portpy MetaData</title></head>
                      <link rel="stylesheet" type="text/css" href="{style}"/>
                      <body>
                      <h1> PortPy Metadata </h1> 
                      <h4> Beams Metadata </h4>
                        {table_1}
                      <h4> Structures Metadata </h4>
                        {table_2}
                      </body>
                    </html>.
                    '''  # create html body and append table to it
            with open('temp.html', 'w') as f:
                f.write(html_string.format(table_1=beams_df.to_html(index=False, header=True, classes='mystyle'),
                                           table_2=struct_df.to_html(index=False, header=True, classes='mystyle'),
                                           style=style_file))
            webbrowser.open('file://' + os.path.realpath('temp.html'))
        else:
            print('Beams table..')
            print(tabulate(beams_df, headers='keys', tablefmt='psql'))  # print the table in console using tabulate
            print('\n\nStructures table..')
            print(tabulate(struct_df, headers='keys', tablefmt='psql'))

    @staticmethod
    def display_patients(data_dir: str = None, in_browser: bool = False) -> None:
        """
        Displays the list of patients included in data_dir folder

        :param data_dir: folder including patient data.
            If it is None, then it assumes the data is in the current directory under sub-folder named "Data"
        :param in_browser: visualize in pretty way in browser. default to False. If false, plot table in console
        :raises invalid directory error: raises an exception if invalid data directory.

        :return display patient information in table

        """

        display_dict = {}  # we add all the relevant information from meta_data to this dictionary
        if data_dir is None:  # if data directory not provided, then use the subfolder named "Data" in the current directory
            data_dir = os.path.join(Path(__file__).parents[1], 'Data')
        if not os.path.exists(data_dir):  # check if valid directory
            raise Exception("Invalid data directory. Please input valid directory")
        pat_ids = os.listdir(data_dir)
        for i, pat_id in enumerate(pat_ids):  # Loop through patients in path
            if "Patient" in pat_id:  # ignore irrelevant folders
                display_dict.setdefault('patient_id', []).append(pat_id)
                meta_data = load_metadata(os.path.join(data_dir, pat_id))  # load metadata for the patients
                # set the keys and append to display dict
                display_dict.setdefault('disease_site', []).append(meta_data['clinical_criteria']['disease_site'])
                ind = meta_data['structures']['name'].index('PTV')
                display_dict.setdefault('ptv_vol_cc', []).append(meta_data['structures']['volume_cc'][ind])
                display_dict.setdefault('num_beams', []).append(len(meta_data['beams_dict']['ID']))
                # check if all the iso centers are same for beams_dict
                res = all(
                    ele == meta_data['beams_dict']['iso_center'][0] for ele in meta_data['beams_dict']['iso_center'])
                if res:
                    display_dict.setdefault('iso_center_shift ', []).append('No')
                else:
                    display_dict.setdefault('iso_center_shift ', []).append('Yes')
        df = pd.DataFrame.from_dict(display_dict)  # convert dictionary to dataframe
        if in_browser:
            style_file = os.path.join(Path(__file__).parents[1], 'df_style.css')  # get style file path
            html_string = '''
                    <html>
                      <head><title>Portpy MetaData</title></head>
                      <link rel="stylesheet" type="text/css" href="{style}"/>
                      <body>
                      <h4> Patients Metadata </h4>
                        {table}
                      </body>
                    </html>.
                    '''  # create html body and append table to it
            with open('temp.html', 'w') as f:
                f.write(
                    html_string.format(table=df.to_html(index=False, header=True, classes='mystyle'), style=style_file))
            webbrowser.open('file://' + os.path.realpath('temp.html'))
        else:
            print(tabulate(df, headers='keys', tablefmt='psql'))  # print in console using tabulate

    @staticmethod
    def plan_metrics(my_plan: Plan, sol: dict) -> None:
        """
        Visualize the plan metrics for clinical criteria in browser.
        It evaluate the plan by comparing the metrics against required criteria.

        If plan value is green color. It meets all the Limits and Goals
        If plan value is yellow color. It meets limits but not goals
        If plan value is red color. It violates both limit and goals

        :param my_plan: object of class Plan
        :param sol: optimal solution dictionary
        :return: plan metrics in browser
        """

        dose_1d = sol['inf_matrix'].A * sol['optimal_intensity'] * my_plan.get_num_of_fractions()

        # convert clinical criteria in dataframe
        df = pd.DataFrame.from_dict(my_plan.clinical_criteria.clinical_criteria_dict['criteria'])
        for ind in range(len(df)):  # Loop through the clinical criteria
            if df.name[ind] == 'max_dose':
                struct = df.parameters[ind]['structure_name']
                max_dose = Evaluation.get_max_dose(sol, dose_1d=dose_1d, struct=struct)  # get max dose_1d
                if 'limit_dose_gy' in df.constraints[ind] or 'goal_dose_gy' in df.constraints[ind]:
                    df.at[ind, 'Plan Value'] = max_dose
                elif 'limit_dose_perc' in df.constraints[ind] or 'goal_dose_perc' in df.constraints[ind]:
                    df.at[ind, 'Plan Value'] = max_dose / (
                            my_plan.get_prescription() * my_plan.get_num_of_fractions()) * 100
            if df.name[ind] == 'mean_dose':
                struct = df.parameters[ind]['structure_name']
                mean_dose = Evaluation.get_mean_dose(sol, dose_1d=dose_1d, struct=struct)
                df.at[ind, 'Plan Value'] = mean_dose
            if df.name[ind] == 'dose_volume_V':
                struct = df.parameters[ind]['structure_name']
                if 'limit_volume_perc' in df.constraints[ind] or 'goal_volume_perc' in df.constraints[ind]:
                    dose = df.parameters[ind]['dose_gy']
                    volume = Evaluation.get_volume(sol, dose_1d=dose_1d, struct=struct, dose_value_gy=dose)
                    df.at[ind, 'Plan Value'] = volume
                elif 'limit_volume_cc' in df.constraints[ind] or 'goal_volume_cc' in df.constraints[ind]:
                    dose = df.parameters[ind]['dose_gy']
                    volume = Evaluation.get_volume(sol, dose_1d=dose_1d, struct=struct, dose_value_gy=dose)
                    vol_cc = my_plan.structures.get_volume_cc(structure_name=struct) * volume / 100
                    df.at[ind, 'Plan Value'] = vol_cc

        def color_plan_value(row):

            highlight_red = 'background-color: red;'
            highlight_green = 'background-color: green;'
            highlight_yellow = 'background-color: yellow;'
            default = ''

            def matching_keys(dictionary, search_string):  # method to find the key of only part of the key match
                """

                :param dictionary: dictionary to search the string
                :param search_string: string to be searched in dictionary
                :return: full key if found search string found else empty string
                """
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

            row_color = [default, default]  # default color for all rows initially
            # must return one string per cell in this row
            if limit_key in row['constraints']:
                if row['Plan Value'] > row['constraints'][limit_key]:
                    row_color = [highlight_red, default]  # make plan value in red
                else:
                    row_color = [highlight_green, default]
            if goal_key in row['constraints']:
                if row['Plan Value'] > row['constraints'][goal_key]:
                    row_color = [highlight_yellow, default]
                else:
                    row_color = [highlight_green, default]
            return row_color

        styled_df = df.style.apply(color_plan_value, subset=['Plan Value', 'constraints'], axis=1)  # apply
        # color to dataframe using df.style method
        html = styled_df.render()  # render to html
        html_string = '''
                        <html>
                          <head><title>Portpy Clinical Criteria Evaluation</title></head>
                          <style> 
                            table, th, td {{font-size:10pt; border:1px solid black; border-collapse:collapse; text-align:left;}}
                            th, td {{padding: 5px;}}
                          </style>
                          <body>
                          <h1> Clinical Criteria</h1>
                          <h4 style="color: green">Meets limit and goal</h4>
                          <h4 style="color: yellow">Meets limit but not goal</h4>
                          <h4 style="color: red">Violate both limit and goal</h4>
                            {table}
                          </body>
                        </html>.
                        '''
        with open('temp.html', 'w') as f:
            f.write(html_string.format(table=html))
        webbrowser.open('file://' + os.path.realpath('temp.html'))
