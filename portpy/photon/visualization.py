from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage import measure
from portpy.photon.evaluation import Evaluation
from matplotlib.lines import Line2D
import os
from portpy.photon.utils import view_in_slicer_jupyter
from pathlib import Path
from typing import List, TYPE_CHECKING
from .ct import CT
from .structures import Structures

if TYPE_CHECKING:
    from portpy.photon.plan import Plan
    from portpy.photon.influence_matrix import InfluenceMatrix
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Visualization:
    dose_type = Literal["Absolute(Gy)", "Relative(%)"]
    volume_type = Literal["Absolute(cc)", "Relative(%)"]

    @staticmethod
    def plot_dvh(my_plan: Plan, sol: dict = None, dose_1d: np.ndarray = None, struct_names: List[str] = None,
                 dose_scale: dose_type = "Absolute(Gy)",
                 volume_scale: volume_type = "Relative(%)", **options):
        """
        Create dvh plot for the selected structures

        :param my_plan: object of class Plan
        :param sol: optimal sol dictionary
        :param dose_1d: dose_1d in 1d voxels
        :param struct_names: structures to be included in dvh plot
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
        >>> Visualization.plot_dvh(my_plan, sol=sol, struct_names=['PTV', 'ESOPHAGUS'], dose_scale='Absolute(Gy)',volume_scale="Relative(%)", show=False, create_fig=True )
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
        show = options['show'] if 'show' in options else False
        # create_fig = options['create_fig'] if 'create_fig' in options else False
        show_criteria = options['show_criteria'] if 'show_criteria' in options else None
        ax = options['ax'] if 'ax' in options else None

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
        if struct_names is None:
            # orgs = []
            struct_names = my_plan.structures.structures_dict['name']
        max_dose = 0.0
        max_vol = 0.0
        all_orgs = my_plan.structures.structures_dict['name']
        # orgs = [struct.upper for struct in orgs]
        pres = my_plan.get_prescription()
        legend = []

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if norm_flag:
            norm_factor = Evaluation.get_dose(sol, dose_1d=dose_1d, struct=norm_struct, volume_per=norm_volume) / pres
            dose_1d = dose_1d / norm_factor
        for i in range(np.size(all_orgs)):
            if all_orgs[i] not in struct_names:
                continue
            # for dose_1d in dose_list:
            #
            x, y = Evaluation.get_dvh(sol, struct=all_orgs[i], dose_1d=dose_1d)
            if dose_scale == 'Absolute(Gy)':
                max_dose = np.maximum(max_dose, x[-1])
                ax.set_xlabel('Dose (Gy)')
            elif dose_scale == 'Relative(%)':
                x = x / pres * 100
                max_dose = np.maximum(max_dose, x[-1])
                ax.set_xlabel('Dose (%)')

            if volume_scale == 'Absolute(cc)':
                y = y * my_plan.structures.get_volume_cc(all_orgs[i]) / 100
                max_vol = np.maximum(max_vol, y[1] * 100)
                ax.set_ylabel('Volume (cc)')
            elif volume_scale == 'Relative(%)':
                max_vol = np.maximum(max_vol, y[0] * 100)
                ax.set_ylabel('Volume Fraction (%)')
            ax.plot(x, 100 * y, linestyle=style, linewidth=width, color=colors[i])
            legend.append(all_orgs[i])

        if show_criteria is not None:
            for s in range(len(show_criteria)):
                if 'dose_volume' in show_criteria[s]['name']:
                    x = show_criteria[s]['parameters']['dose_gy']
                    y = show_criteria[s]['constraints']['limit_volume_perc']
                    ax.plot(x, y, marker='x', color='red', markersize=20)
        # plt.xlabel('Dose (Gy)')
        # plt.ylabel('Volume Fraction (%)')
        ax.set_xlim(0, max_dose * 1.1)
        ax.set_ylim(0, max_vol)
        ax.legend(legend, prop={'size': legend_font_size}, loc="upper right")
        ax.grid(visible=True, which='major', color='#666666', linestyle='-')

        # Show the minor grid lines with very faint and almost transparent grey lines
        # plt.minorticks_on()
        ax.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        y = np.arange(0, 101)
        # if norm_flag:
        #     x = pres * np.ones_like(y)
        # else:
        if dose_scale == "Absolute(Gy)":
            x = pres * np.ones_like(y)
        else:
            x = 100 * np.ones_like(y)

        ax.plot(x, y, color='black')
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        return ax

    @staticmethod
    def plot_fluence_2d(beam_id: int, sol: dict = None, optimal_fluence_2d: List[np.ndarray] = None,
                        inf_matrix: InfluenceMatrix = None, **options):
        """

        Displays fluence in 2d for the given beam_id

        :param beam_id: beam_id of the beam
        :param sol: solution dictionary after optimization
        :param optimal_fluence_2d: List of optimal fluence for all the beams
        :param inf_matrix: Optional. Object of class influence matrix
        :return: 2d optimal fluence plot

        :Example:
        >>> Visualization.plot_fluence_2d(beam_id=0, sol=sol, **options)
        """

        if inf_matrix is None:
            inf_matrix = sol['inf_matrix']

        # getting options_fig:
        figsize = options['figsize'] if 'figsize' in options else (8, 8)
        title = options['title'] if 'title' in options else None
        filename = options['filename'] if 'filename' in options else None
        show = options['show'] if 'show' in options else False
        ax = options['ax'] if 'ax' in options else None

        ind = [i for i in range(len(inf_matrix.beamlets_dict)) if inf_matrix.beamlets_dict[i]['beam_id'] == beam_id]
        if len(ind) == 0:
            raise IndexError('invalid beam id {}'.format(beam_id))
        if sol is not None:
            optimal_fluence_2d = inf_matrix.fluence_1d_to_2d(sol=sol)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        mat = ax.matshow(optimal_fluence_2d[ind[0]])
        ax.set_xlabel('x-axis (beamlets column)')
        ax.set_ylabel('y-axis (beamlets row)')
        plt.colorbar(mat, ax=ax)
        if title is not None:
            ax.set_title('{}'.format(title))
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        return ax

    @staticmethod
    def plot_fluence_3d(beam_id: int, sol: dict = None, optimal_fluence_2d: List[np.ndarray] = None,
                        inf_matrix: InfluenceMatrix = None, **options):
        """
        Displays fluence in 3d for the given beam_id

        :param sol: solution after optimization
        :param beam_id: beam_id of the beam
        :param optimal_fluence_2d: List of optimal fluence for all the beams
        :param inf_matrix: Optional. Object of class influence matrix
        :return: 3d optimal fluence plot

        :Example:
        >>> Visualization.plot_fluence_3d(beam_id=0, sol=sol, **options)
        """
        if inf_matrix is None:
            inf_matrix = sol['inf_matrix']
        # getting options_fig:
        figsize = options['figsize'] if 'figsize' in options else (8, 8)
        title = options['title'] if 'title' in options else None
        filename = options['filename'] if 'filename' in options else None
        show = options['show'] if 'show' in options else False
        ax = options['ax'] if 'ax' in options else None

        ind = [i for i in range(len(inf_matrix.beamlets_dict)) if inf_matrix.beamlets_dict[i]['beam_id'] == beam_id]
        if len(ind) == 0:
            raise IndexError('invalid beam id {}'.format(beam_id))
        if sol is not None:
            optimal_fluence_2d = inf_matrix.fluence_1d_to_2d(sol=sol)
        (ax, surf) = Visualization.surface_plot(optimal_fluence_2d[ind[0]], ax=ax, figsize=figsize,
                                                cmap='viridis', edgecolor='black')
        plt.colorbar(surf, ax=ax, pad=0.2)
        ax.set_zlabel('Fluence Intensity')
        ax.set_xlabel('x-axis (beamlets column)')
        ax.set_ylabel('y-axis (beamlets row)')

        if title is not None:
            ax.set_title('{}'.format(title))
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        return ax

    @staticmethod
    def plot_2d_slice(my_plan: Plan = None, sol: dict = None, ct: CT = None, structs: Structures = None,
                      slice_num: int = 40, struct_names: List[str] = None, show_dose: bool = False,
                      show_struct: bool = True, show_isodose: bool = False,
                      **options) -> None:
        """

        Plot 2d view of ct, dose_1d, isodose and struct_name contours

        :param sol: solution to optimization
        :param my_plan: object of class Plan
        :param ct: Optional. object of class CT
        :param structs: Optional. object of class structs
        :param slice_num: slice number
        :param struct_names: structures for which contours to be displayed on the slice view. e.g. struct_names = ['PTV, ESOPHAGUS']
        :param show_dose: view dose_1d on the slice
        :param show_struct: view struct_name on the slice
        :param show_isodose: view isodose
        :param dpi: Default dpi=100 for figure
        :return: plot 2d view of ct, dose_1d, isodose and struct_name contours

        :Example:
        >>> Visualization.plot_2d_slice(my_plan, sol=sol, slice_num=50, struct_names=['PTV'], show_isodose=False)
        """

        # getting options_fig:
        figsize = options['figsize'] if 'figsize' in options else (8, 8)
        title = options['title'] if 'title' in options else None
        filename = options['filename'] if 'filename' in options else None
        dpi = options['dpi'] if 'dpi' in options else 100
        show = options['show'] if 'show' in options else False
        ax = options['ax'] if 'ax' in options else None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        plt.rcParams["figure.autolayout"] = True
        if ct is None:
            ct = my_plan.ct
        ct_hu_3d = ct.ct_dict['ct_hu_3d'][0]
        ax.imshow(ct_hu_3d[slice_num, :, :], cmap='gray')

        # adjust the main plot to make room for the legends
        plt.subplots_adjust(left=0.2)
        dose_3d = []
        if sol is not None:
            show_dose = True
        if show_dose:
            dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'] * my_plan.get_num_of_fractions())
            dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
            masked = np.ma.masked_where(dose_3d[slice_num, :, :] <= 0, dose_3d[slice_num, :, :])
            im = ax.imshow(masked, alpha=0.4, interpolation='none',
                           cmap='rainbow')

            plt.colorbar(im, ax=ax, pad=0.1)

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
        if title is not None:
            ax.set_title('{}: Axial View - Slice #: {}'.format(title, slice_num))
        else:
            ax.set_title('Axial View - Slice #: {}'.format(slice_num))

        if show_struct:
            if structs is None:
                structs = my_plan.structures
            if struct_names is None:
                struct_names = structs.structures_dict['name']
            struct_masks = structs.structures_dict['structure_mask_3d']
            all_mask = []
            colors = Visualization.get_colors()
            for i in range(len(struct_names)):
                ind = structs.structures_dict['name'].index(struct_names[i])
                cmap = mpl.colors.ListedColormap(colors[i])
                contours = measure.find_contours(struct_masks[ind][slice_num, :, :], 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=colors[i])
            labels = [struct for struct in struct_names]
            # get the colors of the values, according to the
            import matplotlib.patches as mpatches
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # rax.labels = labels
            ax.legend(handles=patches, bbox_to_anchor=(0.1, 0.8), loc=2, borderaxespad=0.)
            # bbox_transform=fig.transFigure)
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", dpi=300)
        return ax

    @staticmethod
    def view_in_slicer(my_plan: Plan, slicer_path: str = None, data_dir: str = None) -> None:
        """

        :param my_plan: object of class Plan
        :param slicer_path: slicer executable path on your local machine
        :param data_dir: the folder path where data located, defaults to None.
                If path = None, then it assumes the data is in sub-folder named data in the current directory
        :return: plot the images in 3d slicer

        view ct, dose_1d and struct_name set images in 3d slicer

        :Example:
        >>> Visualization.view_in_slicer(my_plan, slicer_path='C:/Slicer/Slicer.exe', data_dir='path/to/nrrd/image')
        """
        if slicer_path is None:
            slicer_path = r'C:\ProgramData\NA-MIC\Slicer 4.11.20210226\Slicer.exe'
        if data_dir is None:
            data_dir = os.path.join('..', 'data', my_plan.patient_id)
        if not os.path.exists(data_dir):  # check if valid directory
            raise Exception("Invalid data directory. Please input valid directory")
        slicer_script_dir = os.path.join(Path(__file__).parents[0], 'utils', 'slicer_script.py')
        import subprocess

        # run python script 'slicer_script.py' in slicer python interface. Currenr limitation segmentation.nrrd file
        # doesnt know the struct_name names. Hence we manually pass struct_name names to python script
        subprocess.run([slicer_path, f"--python-script", slicer_script_dir, data_dir,
                        ','.join(my_plan.structures.structures_dict['name'])], shell=False)
        print('Done')

    @staticmethod
    def view_in_slicer_jupyter(my_plan: Plan, dose_1d: np.ndarray = None, sol: dict = None,
                               ct_name: str = 'ct', dose_name: str = 'dose', struct_set_name: str = 'rt_struct',
                               show_ct: bool = True,
                               show_dose: bool = True,
                               show_structs: bool = True):
        """
        This method helps to visualize CT, Dose and Rt struct in 3d slicer


        :param my_plan: object of class plan
        :param dose_1d: dose in 1d
        :param sol: solution dictionary
        :param ct_name: Default to 'ct'. name of the ct node in 3D slicer
        :param dose_name: Default to 'dose'. name of the dose node in 3D slicer
        :param struct_set_name: name of the rtstruct
        :param show_structs: default to True. If false, will not create struct_name node
        :param show_dose: default to True. If false, will not create dose node
        :param show_ct:default to True. If false, will not create ct node
        :return: visualize in slicer jupyter
        """
        view_in_slicer_jupyter(my_plan, dose_1d=dose_1d, sol=sol, ct_name=ct_name,
                               dose_name=dose_name,
                               struct_set_name=struct_set_name, show_ct=show_ct,
                               show_dose=show_dose,
                               show_structs=show_structs)

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
        colors = ['#ffe119', '#4363d8', '#f58231', '#911eb4',
                  '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
                  '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                  '#000075', '#808080', '#ffffff', '#000000', '#e6194b', '#3cb44b']
        return colors

    @staticmethod
    def surface_plot(matrix: np.ndarray, **kwargs):
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, np.transpose(matrix), **kwargs)
        return ax, surf

    @staticmethod
    def get_cmap_colors(n, name='hsv'):
        """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name."""
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
    def is_notebook() -> bool:
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                if 'google.colab' in str(get_ipython()):
                    return True
                else:
                    return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter
        except:
            return False  # Probably standard Python interpreter

    @staticmethod
    def surface_plot(matrix, ax=None, figsize=(8, 8), **kwargs):
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='3d'))
        surf = ax.plot_surface(x, y, np.transpose(matrix), **kwargs)
        return ax, surf
