from scipy import interpolate
import numpy as np
import pandas as pd
import webbrowser
from typing import List, Union
import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .plan import Plan
from .clinical_criteria import ClinicalCriteria
from tabulate import tabulate


class Evaluation:
    """
    Class for evaluating different metrics for the plan

    - **Methods** ::
        :display_clinical_criteria(my_plan, sol)
            display plan values corresponding to given clinical criteria
        :get_dose(sol, struct, volume_per)
            get dose at the given volume in percentage
        :get_volume(sol: struct dose_value_gy)
            Get volume at dose_1d value in Gy


    """

    @staticmethod
    def get_dose(sol: dict, struct: str, volume_per: float, dose_1d: np.ndarray = None,
                 weight_flag: bool = True) -> float:
        """
        Get dose_1d at volume percentage

        :param sol: solution dictionary
        :param dose_1d: dose_1d in 1d
        :param struct: struct_name name for which to get the dose_1d
        :param volume_per: query the dose_1d at percentage volume
        :param weight_flag: for non uniform voxels weight flag always True
        :return: dose_1d at volume_percentage

        :Example:

        >>> Evaluation.get_dose(sol=sol, struct='PTV', volume_per=90)

        """
        x, y = Evaluation.get_dvh(sol, dose_1d=dose_1d, struct=struct, weight_flag=weight_flag)
        f = interpolate.interp1d(100 * y, x)

        return f(volume_per)

    @staticmethod
    def display_clinical_criteria(my_plan: Plan, sol: Union[dict, List[dict]], html_file_name='temp.html',
                                  sol_names: List[str] = None, clinical_criteria: ClinicalCriteria = None,
                                  return_df: bool = False, in_browser: bool = False):
        """
        Visualization the plan metrics for clinical criteria in browser.
        It evaluate the plan by comparing the metrics against required criteria.

        If plan value is green color. It meets all the Limits and Goals
        If plan value is yellow color. It meets limits but not goals
        If plan value is red color. It violates both limit and goals

        :param my_plan: object of class Plan
        :param sol: optimal solution dictionary
        :param html_file_name:  name of the html file to be launched in browser
        :param sol_names: Default to Plan Value. column names for the plan evaluation
        :param clinical_criteria: clinical criteria to be evaluated
        :param return_df: return df instead of visualization
        :param in_browser: display table in browser
        :return: plan metrics in browser
        """

        # convert clinical criteria in dataframe
        if clinical_criteria is None:
            clinical_criteria = my_plan.clinical_criteria
        df = pd.DataFrame.from_dict(clinical_criteria.clinical_criteria_dict['criteria'])
        if isinstance(sol, dict):
            sol = [sol]
        if sol_names is None:
            if len(sol) > 1:
                sol_names = ['Plan Value ' + str(i) for i in range(len(sol))]
            else:
                sol_names = ['Plan Value']
        for p, s in enumerate(sol):
            dose_1d = s['inf_matrix'].A @ (s['optimal_intensity'] * my_plan.get_num_of_fractions())
            for ind in range(len(df)):  # Loop through the clinical criteria
                if df.name[ind] == 'max_dose':
                    struct = df.parameters[ind]['structure_name']
                    if struct in my_plan.structures.get_structures():
                        max_dose = Evaluation.get_max_dose(s, dose_1d=dose_1d, struct=struct)  # get max dose_1d
                        if 'limit_dose_gy' in df.constraints[ind] or 'goal_dose_gy' in df.constraints[ind]:
                            df.at[ind, sol_names[p]] = max_dose
                        elif 'limit_dose_perc' in df.constraints[ind] or 'goal_dose_perc' in df.constraints[ind]:
                            df.at[ind, sol_names[p]] = max_dose / (
                                    my_plan.get_prescription() * my_plan.get_num_of_fractions()) * 100
                if df.name[ind] == 'mean_dose':
                    struct = df.parameters[ind]['structure_name']
                    if struct in my_plan.structures.get_structures():
                        mean_dose = Evaluation.get_mean_dose(s, dose_1d=dose_1d, struct=struct)
                        df.at[ind, sol_names[p]] = mean_dose
                if df.name[ind] == 'dose_volume_V':
                    struct = df.parameters[ind]['structure_name']
                    if struct in my_plan.structures.get_structures():
                        if 'limit_volume_perc' in df.constraints[ind] or 'goal_volume_perc' in df.constraints[ind]:
                            dose = df.parameters[ind]['dose_gy']
                            volume = Evaluation.get_volume(s, dose_1d=dose_1d, struct=struct, dose_value_gy=dose)
                            df.at[ind, sol_names[p]] = volume
                        elif 'limit_volume_cc' in df.constraints[ind] or 'goal_volume_cc' in df.constraints[ind]:
                            dose = df.parameters[ind]['dose_gy']
                            volume = Evaluation.get_volume(s, dose_1d=dose_1d, struct=struct, dose_value_gy=dose)
                            vol_cc = my_plan.structures.get_volume_cc(structure_name=struct) * volume / 100
                            df.at[ind, sol_names[p]] = vol_cc
        pd.set_option("display.precision", 3)
        df.dropna(axis=0, inplace=True)  # remove structures which are not present
        df.reset_index(drop=True, inplace=True)  # reset the index

        def color_plan_value(row):

            highlight_red = 'background-color: red;'
            highlight_green = 'background-color: green;'
            highlight_orange = 'background-color: orange;'
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

            row_color = len(row) * [default]  # default color for all rows initially
            # must return one string per cell in this row
            for i in range(len(row) - 1):
                if limit_key in row['constraints']:
                    if not (pd.isnull(row[i])):
                        if row[i] > row['constraints'][limit_key] + 0.0001:  # added epsilon to avoid minor differences
                            row_color[i] = highlight_red  # make plan value in red
                        else:
                            row_color[i] = highlight_green  # make plan value in red
                if goal_key in row['constraints']:
                    if not (pd.isnull(row[i])):
                        if row[i] > row['constraints'][goal_key] + 0.0001:
                            row_color[i] = highlight_orange  # make plan value in red
                        else:
                            row_color[i] = highlight_green  # make plan value in red
            return row_color

        sol_names.append('constraints')
        styled_df = df.style.apply(color_plan_value, subset=sol_names, axis=1)  # apply
        # color to dataframe using df.style method
        if return_df:
            return styled_df
        if in_browser:
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
                                                  <h4 style="color: orange">Meets limit but not goal</h4>
                                                  <h4 style="color: red">Violate both limit and goal</h4>
                                                    {table}
                                                  </body>
                                                </html>.
                                                '''
            with open(html_file_name, 'w') as f:
                f.write(html_string.format(table=html))
            webbrowser.open('file://' + os.path.realpath(html_file_name))

        else:
            if Evaluation.is_notebook():
                from IPython.display import display
                with pd.option_context('display.max_rows', None,
                                       'display.max_columns', None,
                                       'display.precision', 3,
                                       ):

                    display(styled_df)
            else:
                print(tabulate(df, headers='keys', tablefmt='psql'))  # print in console using tabulate

    @staticmethod
    def get_volume(sol: dict, struct: str, dose_value_gy: float, dose_1d: np.ndarray = None,
                   weight_flag: bool = True) -> float:
        """
        Get volume at dose_1d value in Gy

        :param sol: solution dictionary
        :param dose_1d: dose_1d in 1d
        :param struct: struct_name name for which to get the dose_1d
        :param dose_value_gy: query the volume at dose_value
        :param weight_flag: for non uniform voxels weight flag always True
        :return: dose_1d at volume_percentage

        :Example:

        >>> Evaluation.get_volume(sol=sol, struct='PTV', dose_value_gy=60)

        """
        x, y = Evaluation.get_dvh(sol, dose_1d=dose_1d, struct=struct, weight_flag=weight_flag)
        x1, indices = np.unique(x, return_index=True)
        y1 = y[indices]
        f = interpolate.interp1d(x1, 100 * y1)
        if dose_value_gy > max(x1):
            print('Warning: dose_1d value {} is greater than max dose_1d for {}'.format(dose_value_gy, struct))
            return 0
        else:
            return f(dose_value_gy)

    @staticmethod
    def get_dvh(sol: dict, struct: str, dose_1d: np.ndarray = None, weight_flag: bool = True):
        """
        Get dvh for the struct_name

        :param sol: optimal solution dictionary
        :param dose_1d: dose_1d which is not in solution dictionary
        :param struct: struct_name name
        :param weight_flag: for non uniform voxels weight flag always True
        :return: x, y --> dvh for the struct_name

        :Example:

        >>> Evaluation.get_dvh(sol=sol, struct='PTV')

        """
        inf_matrix = sol['inf_matrix']
        vox = inf_matrix.get_opt_voxels_idx(struct)
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        org_sort_dose = np.sort(dose_1d[vox])
        sort_ind = np.argsort(dose_1d[vox])
        org_sort_dose = np.append(org_sort_dose, org_sort_dose[-1] + 0.01)
        x = org_sort_dose
        if weight_flag:
            # org_points_sort_spacing = my_plan._structures.opt_voxels_dict['dose_voxel_resolution_XYZ_mm']
            # org_points_sort_volume = org_points_sort_spacing[:, 0] * org_points_sort_spacing[:,
            #                                                          1] * org_points_sort_spacing[:, 2]
            # sum_weight = np.sum(org_points_sort_volume)
            org_weights = inf_matrix.get_opt_voxels_volume_cc(struct)
            org_sort_weights = org_weights[sort_ind]
            sum_weight = np.sum(org_sort_weights)
            y = [1]
            for j in range(len(org_sort_weights)):
                y.append(y[-1] - org_sort_weights[j] / sum_weight)
        else:
            y = np.ones(len(vox) + 1) - np.arange(0, len(vox) + 1) / len(vox)
        y[-1] = 0
        y = np.array(y)
        return x, y

    @staticmethod
    def get_max_dose(sol: dict, struct: str, dose_1d=None) -> float:
        """
        Get maximum dose_1d for the struct_name

        :param sol: optimal solution dictionary
        :param dose_1d: dose_1d which is not in solution dictionary
        :param struct: struct_name name

        :return: maximum dose_1d for the struct_name
        """
        inf_matrix = sol['inf_matrix']
        vox = inf_matrix.get_opt_voxels_idx(struct)
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        return np.max(dose_1d[vox])

    @staticmethod
    def get_mean_dose(sol: dict, struct: str, dose_1d=None) -> np.ndarray:
        """
                Get mean dose_1d for the struct_name

                :param sol: optimal solution dictionary
                :param dose_1d: dose_1d which is not in solution dictionary
                :param struct: struct_name name

                :return: mean dose_1d for the struct_name
                """
        inf_matrix = sol['inf_matrix']
        vox = inf_matrix.get_opt_voxels_idx(struct)
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        return np.mean(dose_1d[vox])

    @staticmethod
    def get_BED(my_plan: Plan, sol: dict = None, dose_per_fraction_1d: np.ndarray = None, alpha=1, beta=1) -> np.ndarray:
        """
        Get Biologically equivalent dose (BED) for the struct_name

        :param my_plan: Object of class Plan
        :param sol: optimal solution dictionary
        :param dose_per_fraction_1d: dose_1d which is not in solution dictionary

        :return: BED
        """

        if dose_per_fraction_1d is None:
            dose_per_fraction_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'])
        bed_d = np.zeros_like(dose_per_fraction_1d)
        for i in range(int(my_plan.get_num_of_fractions())):
            bed_d = bed_d + (dose_per_fraction_1d + (dose_per_fraction_1d**2/(alpha/beta)))
        return bed_d

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
