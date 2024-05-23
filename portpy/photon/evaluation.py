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
    def display_clinical_criteria(my_plan: Plan, sol: Union[dict, List[dict]] = None, dose_1d: Union[np.ndarray, List[np.ndarray]]=None, html_file_name='temp.html',
                                  sol_names: List[str] = None, clinical_criteria: ClinicalCriteria = None,
                                  return_df: bool = False, in_browser: bool = False, path: str = None, open_browser: bool = True):
        """
        Visualization the plan metrics for clinical criteria in browser.
        It evaluate the plan by comparing the metrics against required criteria.

        If plan value is green color. It meets all the Limits and Goals
        If plan value is yellow color. It meets limits but not goals
        If plan value is red color. It violates both limit and goals

        :param my_plan: object of class Plan
        :param sol: optimal solution dictionary
        :param dose_1d: vectorized dose 1d array
        :param html_file_name:  name of the html file to be launched in browser
        :param sol_names: Default to Plan Value. column names for the plan evaluation
        :param clinical_criteria: clinical criteria to be evaluated
        :param return_df: return df instead of visualization
        :param in_browser: display table in browser
        :param path: path for saving the html file which opens up in browser
        :param open_browser: if true, html will be launched in browser
        :return: plan metrics in browser
        """
        import re
        # convert clinical criteria in dataframe
        if clinical_criteria is None:
            clinical_criteria = my_plan.clinical_criteria
        # df = pd.DataFrame.from_dict(clinical_criteria.clinical_criteria_dict['criteria'])
        df = pd.json_normalize(clinical_criteria.clinical_criteria_dict['criteria'])
        dose_volume_V_ind = df.index[df['type'] == 'dose_volume_V'].tolist()
        if dose_volume_V_ind:
            volumn_cols = [col for col in df.columns if 'volume' in col]
            if volumn_cols:
                perc_col = [col_name for col_name in volumn_cols if 'perc' in col_name]
                cc_col = [col_name for col_name in volumn_cols if 'cc' in col_name]
                for col in perc_col:
                    if 'limit' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'Limit', col, '%')
                    if 'goal' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'Goal', col, '%')
                for col in cc_col:
                    if 'limit' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'Limit', col, 'cc')
                    if 'goal' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'Goal', col, 'cc')
            dose_cols = [col for col in df.columns if 'parameters.dose' in col]
            if dose_cols:
                for col in dose_cols:
                    if 'perc' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'type', col, '%', 'dose_volume_V')
                    if 'gy' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'type', col, 'Gy', 'dose_volume_V')

        dose_cols = [col for col in df.columns if 'dose' in col]
        if dose_cols:
            perc_col = [col_name for col_name in dose_cols if 'perc' in col_name]
            gy_col = [col_name for col_name in dose_cols if 'gy' in col_name]
            for col in perc_col:
                if 'limit' in col:
                    df = Evaluation.add_dvh_to_frame(my_plan, df, 'Limit', col, '%')
                if 'goal' in col:
                    df = Evaluation.add_dvh_to_frame(my_plan, df, 'Goal', col, '%')
            for col in gy_col:
                if 'limit' in col:
                    df = Evaluation.add_dvh_to_frame(my_plan, df, 'Limit', col, 'Gy')
                if 'goal' in col:
                    df = Evaluation.add_dvh_to_frame(my_plan, df, 'Goal', col, 'Gy')

        dose_volume_D_ind = df.index[df['type'] == 'dose_volume_D'].tolist()
        if dose_volume_D_ind:
            vol_cols = [col for col in df.columns if 'parameters.volume' in col]
            if vol_cols:
                for col in vol_cols:
                    if 'perc' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'type', col, '%', 'dose_volume_D')
                    if 'cc' in col:
                        df = Evaluation.add_dvh_to_frame(my_plan, df, 'type', col, 'cc', 'dose_volume_D')

        # refine df
        df = df.rename(columns={'parameters.structure_name': 'structure_name', 'type': 'constraint'})
        # df = df.drop(
        #     ['parameters.dose_gy', 'constraints.limit_dose_gy', 'constraints.limit_volume_perc',
        #      'constraints.goal_dose_gy', 'constraints.goal_volume_perc','parameters.structure_def'], axis=1, errors='ignore')
        if 'Goal' not in df:
            df['Goal'] = ''
        df = df[['constraint', 'structure_name', 'Limit', 'Goal']]

        dose_1d_list = []
        dummy_sol = {}
        if isinstance(sol, dict):
            sol = [sol]
        if dose_1d is None:
            for p, s in enumerate(sol):
                dose_1d_list.append(s['inf_matrix'].A @ (s['optimal_intensity'] * my_plan.get_num_of_fractions()))
        else:
            if isinstance(dose_1d, np.ndarray):
                dose_1d_list = [dose_1d]
            else:
                dose_1d_list = dose_1d
        if sol_names is None:
            if len(dose_1d_list) > 1:
                sol_names = ['Plan Value ' + str(i) for i in range(len(dose_1d_list))]
            else:
                sol_names = ['Plan Value']
        for p, dose_1d in enumerate(dose_1d_list):
            dummy_sol['inf_matrix'] = my_plan.inf_matrix
            dummy_sol['dose_1d'] = dose_1d
            for ind in range(len(df)):  # Loop through the clinical criteria
                if df.constraint[ind] == 'max_dose':
                    struct = df.structure_name[ind]
                    if struct in my_plan.structures.get_structures():

                        max_dose = Evaluation.get_max_dose(dummy_sol, dose_1d=dose_1d, struct=struct)  # get max dose_1d
                        if 'Gy' in str(df.Limit[ind]) or 'Gy' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(max_dose,2)
                        elif '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(max_dose / my_plan.get_prescription() * 100, 2)
                elif df.constraint[ind] == 'mean_dose':
                    struct = df.structure_name[ind]
                    if struct in my_plan.structures.get_structures():
                        mean_dose = Evaluation.get_mean_dose(dummy_sol, dose_1d=dose_1d, struct=struct)
                        if 'Gy' in str(df.Limit[ind]) or 'Gy' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(mean_dose, 2)
                        elif '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(mean_dose / my_plan.get_prescription() * 100, 2)
                elif "V(" in df.constraint[ind]:
                    struct = df.structure_name[ind]
                    if struct in my_plan.structures.get_structures():
                        dose = re.findall(r"[-+]?(?:\d*\.*\d+)", df.constraint[ind])[0]
                        # convert dose to Gy
                        if '%' in df.constraint[ind]:
                            dose = float(dose)
                            dose = dose * my_plan.get_prescription() / 100
                        elif 'Gy' in df.constraint[ind]:
                            dose = float(dose)
                        # get volume in perc
                        volume = Evaluation.get_volume(dummy_sol, dose_1d=dose_1d, struct=struct, dose_value_gy=dose)
                        if '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]): # we are writing str since nan values throws error
                            df.at[ind, sol_names[p]] = np.round(volume, 2)
                        elif 'cc' in str(df.Limit[ind]) or 'cc' in str(df.Goal[ind]):
                            vol_cc = my_plan.structures.get_volume_cc(structure_name=struct) * volume / 100
                            df.at[ind, sol_names[p]] = np.round(vol_cc, 2)
                elif "D(" in df.constraint[ind]:
                    struct = df.structure_name[ind]
                    if struct in my_plan.structures.get_structures():
                        volume = re.findall(r"[-+]?(?:\d*\.*\d+)", df.constraint[ind])[0]
                        # convert volume to perc
                        if '%' in df.constraint[ind]:
                            volume = float(volume)
                        elif 'cc' in df.constraint[ind]:
                            volume = float(volume)
                            volume = volume / my_plan.structures.get_volume_cc(structure_name=struct) * 100

                        # get dose
                        dose = Evaluation.get_dose(dummy_sol, dose_1d=dose_1d, struct=struct, volume_per=volume)
                        if '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]): # we are writing str since nan values throws error
                            df.at[ind, sol_names[p]] = np.round(dose/my_plan.get_prescription()*100, 2)
                        elif 'Gy' in str(df.Limit[ind]) or 'Gy' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(dose, 2)
        df.round(2)
        df = df[df[sol_names].notna().all(axis=1)]  # remove rows for which plan value is Nan
        df = df.fillna('')
        # df.dropna(axis=0, inplace=True)  # remove structures which are not present
        # df.reset_index(drop=True, inplace=True)  # reset the index

        def color_plan_value(row):

            highlight_red = 'background-color: red;'   # red
            highlight_green = 'background-color: #90ee90'  # green
            highlight_orange = 'background-color: #ffb38a'  # orange
            default = ''

            row_color = len(row) * [default]  # default color for all rows initially
            # must return one string per cell in this row
            for i in range(len(row) - 2):
                if 'Limit' in row:
                    if not row['Limit'] == '':
                        limit = float(re.findall(r"[-+]?(?:\d*\.*\d+)", row['Limit'])[0])
                        if row[i] > limit + 0.0001:  # added epsilon to avoid minor differences
                            row_color[i] = highlight_red  # make plan value in red
                        else:
                            row_color[i] = highlight_green  # make plan value in red
                if 'Goal' in row:
                    if not row['Goal'] == '':
                        goal = float(re.findall(r"[-+]?(?:\d*\.*\d+)", row['Goal'])[0])
                        if row[i] > goal + 0.0001:
                            row_color[i] = highlight_orange  # make plan value in red
                        else:
                            row_color[i] = highlight_green  # make plan value in red
            return row_color

        sol_names.append('Limit')
        sol_names.append('Goal')
        df.style.set_properties(**{'text-align': 'right'})
        styled_df = df.style.apply(color_plan_value, subset=sol_names, axis=1)  # apply
        styled_df.set_properties(**{'text-align': 'center'}).format(precision=2)

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
                                                      <h4 style="color: #90ee90">Meets limit and goal</h4>
                                                      <h4 style="color: #ffb38a">Meets limit but not goal</h4>
                                                      <h4 style="color: red">Violate both limit and goal</h4>
                                                        {table}
                                                      </body>
                                                    </html>.
                                                    '''
            if path is None:
                path = os.getcwd()
            html_file_path = os.path.join(path, html_file_name)
            with open(html_file_path, 'w') as f:
                f.write(html_string.format(table=html))
            if open_browser:
                webbrowser.open('file://' + os.path.realpath(html_file_path))

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
        if np.array_equal(x, np.array([0])) and np.array_equal(y, np.array([0])):
            return 0
        f = interpolate.interp1d(100 * y, x)
        if volume_per > 100.1:
            print('Warning: Volume Percentage: {} for structure {} is invalid'.format(volume_per, struct))
            return 0
        return f(volume_per)

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
        if np.array_equal(x, np.array([0])) and np.array_equal(y, np.array([0])):
            return 0
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
        if len(vox) == 0:
            return np.array([0]), np.array([0])  # bug fix. if single 0 it can throw error while doing interpolation
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
            frac_vol = inf_matrix.get_fraction_of_vol_in_calc_box(struct)
            if frac_vol is None:
                y = [1]
            else:
                y = [frac_vol]
            for j in range(len(org_sort_weights)):
                y.append(y[-1] - (org_sort_weights[j] / sum_weight)*frac_vol)
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
        if len(vox) == 0:
            return 0
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        return np.max(dose_1d[vox])

    @staticmethod
    def get_mean_dose(sol: dict, struct: str, dose_1d=None) -> float:
        """
                Get mean dose_1d for the struct_name

                :param sol: optimal solution dictionary
                :param dose_1d: dose_1d which is not in solution dictionary
                :param struct: struct_name name

                :return: mean dose_1d for the struct_name
                """
        inf_matrix = sol['inf_matrix']
        vox = inf_matrix.get_opt_voxels_idx(struct)
        if len(vox) == 0:
            return np.array(0)
        if dose_1d is None:
            dose_1d = sol['dose_1d']
        frac_vol = inf_matrix.get_fraction_of_vol_in_calc_box(struct)
        mean_dose = (1 / sum(inf_matrix.get_opt_voxels_volume_cc(struct))) * (np.sum((np.multiply(inf_matrix.get_opt_voxels_volume_cc(struct), dose_1d[vox]))))
        return mean_dose * frac_vol

    @staticmethod
    def get_conformity_index(my_plan: Plan, sol: dict = None, dose_3d: np.ndarray = None, target_structure='PTV') -> float:
        """
        Calculate conformity index for the dose
        Closer to 1 is more better

        :param my_plan: object of class Plan
        :param sol: optimal solution dictionary
        :param dose_3d: dose in 3d array
        :param target_structure: target structure name

        :return: paddick conformity index

        """
        # calulating paddick conformity index
        percentile = 0.95  # reference isodose
        pres = my_plan.get_prescription()
        if dose_3d is None:
            dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'] * my_plan.get_num_of_fractions())
            dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
        pres_iso_dose_mask = (dose_3d >= pres * percentile).astype(int)
        V_iso_pres = np.count_nonzero(pres_iso_dose_mask)
        ptv_mask = my_plan.structures.get_structure_mask_3d(target_structure)
        V_ptv = np.count_nonzero(ptv_mask)
        V_pres_iso_ptv = np.count_nonzero(pres_iso_dose_mask * ptv_mask)
        conformity_index = V_pres_iso_ptv * V_pres_iso_ptv / (V_ptv * V_iso_pres)
        return conformity_index

    @staticmethod
    def get_homogeneity_index(my_plan: Plan, sol: dict = None, dose_3d: np.ndarray = None, target_structure='PTV') -> float:
        """
                Calculate homogeneity index for the dose
                Closer to 0 is more better

                :param my_plan: object of class Plan
                :param sol: optimal solution dictionary
                :param dose_3d: dose in 3d array
                :param target_structure: target structure name

                :return: homogeneity index

                """
        if dose_3d is None:
            dose_1d = sol['inf_matrix'].A @ (sol['optimal_intensity'] * my_plan.get_num_of_fractions())
            dose_3d = sol['inf_matrix'].dose_1d_to_3d(dose_1d=dose_1d)
        ptv = my_plan.structures.get_structure_mask_3d(target_structure)
        ptv_dose = dose_3d[np.where(ptv == 1)]
        PTV_D2 = np.percentile(ptv_dose, 98)
        PTV_D50 = np.percentile(ptv_dose, 50)
        PTV_D98 = np.percentile(ptv_dose, 2)
        return (PTV_D2 - PTV_D98) / PTV_D50

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
    def add_dvh_to_frame(my_plan: Plan, df: pd.DataFrame, new_column_name: str, old_column_name: str, unit: str, dvh_type=None):
        req_ind = df.index[~df[old_column_name].isnull()].tolist()
        for ind in req_ind:
            if dvh_type is None:
                df.loc[ind, new_column_name] = str(round(Evaluation.get_num(my_plan, df[old_column_name][ind]), 2)) + unit
            elif dvh_type is not None:
                if dvh_type == 'dose_volume_V':
                    df.loc[ind, new_column_name] = 'V(' + str(round(Evaluation.get_num(my_plan, df[old_column_name][ind]), 2)) + unit + ')'
                elif dvh_type == 'dose_volume_D':
                    df.loc[ind, new_column_name] = 'D(' + str(round(Evaluation.get_num(my_plan, df[old_column_name][ind]), 2)) + unit + ')'
        return df

    @staticmethod
    def get_num(my_plan, string: Union[str, float]):
        if "prescription_gy" in str(string):
            prescription_gy = my_plan.get_prescription()
            return eval(string)
        elif isinstance(string, float) or isinstance(string, int):
            return string
        else:
            raise Exception('Invalid constraint')
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
