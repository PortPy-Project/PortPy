# Copyright 2025, the PortPy Authors
#
# Licensed under the Apache License, Version 2.0 with the Commons Clause restriction.
# You may obtain a copy of the Apache 2 License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ----------------------------------------------------------------------
# Commons Clause Restriction Notice:
# PortPy is licensed under Apache 2.0 with the Commons Clause.
# You may use, modify, and share the code for non-commercial
# academic and research purposes only.
# Commercial use — including offering PortPy as a service,
# or incorporating it into a commercial product — requires
# a separate commercial license.
# ----------------------------------------------------------------------

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
        if df.empty:
            dose_volume_V_ind = []
        else:
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

        if df.empty:
            dose_volume_D_ind = []
        else:
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
        df = df.rename(columns={'parameters.structure_name': 'Structure Name', 'type': 'Constraint'})
        # df = df.drop(
        #     ['parameters.dose_gy', 'constraints.limit_dose_gy', 'constraints.limit_volume_perc',
        #      'constraints.goal_dose_gy', 'constraints.goal_volume_perc','parameters.structure_def'], axis=1, errors='ignore')
        for label in ['Constraint', 'Structure Name', 'Limit', 'Goal']:
            if label not in df:
                df[label] = ''
        # if 'Goal' not in df:
        #     df['Goal'] = ''
        df = df[['Constraint', 'Structure Name', 'Limit', 'Goal']]

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
                if df.Constraint[ind] == 'max_dose':
                    struct = df.at[ind, 'Structure Name']
                    if struct in my_plan.structures.get_structures():

                        max_dose = Evaluation.get_max_dose(dummy_sol, dose_1d=dose_1d, struct=struct)  # get max dose_1d
                        if 'Gy' in str(df.Limit[ind]) or 'Gy' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(max_dose,2)
                        elif '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(max_dose / my_plan.get_prescription() * 100, 2)
                elif df.Constraint[ind] == 'mean_dose':
                    struct = df.at[ind, 'Structure Name']
                    if struct in my_plan.structures.get_structures():
                        mean_dose = Evaluation.get_mean_dose(dummy_sol, dose_1d=dose_1d, struct=struct)
                        if 'Gy' in str(df.Limit[ind]) or 'Gy' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(mean_dose, 2)
                        elif '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(mean_dose / my_plan.get_prescription() * 100, 2)
                elif "V(" in df.Constraint[ind]:
                    struct = df.at[ind, 'Structure Name']
                    if struct in my_plan.structures.get_structures():
                        dose = re.findall(r"[-+]?(?:\d*\.*\d+)", df.Constraint[ind])[0]
                        # convert dose to Gy
                        if '%' in df.Constraint[ind]:
                            dose = float(dose)
                            dose = dose * my_plan.get_prescription() / 100
                        elif 'Gy' in df.Constraint[ind]:
                            dose = float(dose)
                        # get volume in perc
                        volume = Evaluation.get_volume(dummy_sol, dose_1d=dose_1d, struct=struct, dose_value_gy=dose)
                        if '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]): # we are writing str since nan values throws error
                            df.at[ind, sol_names[p]] = np.round(volume, 2)
                        elif 'cc' in str(df.Limit[ind]) or 'cc' in str(df.Goal[ind]):
                            vol_cc = my_plan.structures.get_volume_cc(structure_name=struct) * volume / 100
                            df.at[ind, sol_names[p]] = np.round(vol_cc, 2)
                elif "D(" in df.Constraint[ind]:
                    struct = df.at[ind, 'Structure Name']
                    if struct in my_plan.structures.get_structures():
                        volume = re.findall(r"[-+]?(?:\d*\.*\d+)", df.Constraint[ind])[0]
                        # convert volume to perc
                        if '%' in df.Constraint[ind]:
                            volume = float(volume)
                        elif 'cc' in df.Constraint[ind]:
                            volume = float(volume)
                            volume = volume / my_plan.structures.get_volume_cc(structure_name=struct) * 100

                        # get dose
                        dose = Evaluation.get_dose(dummy_sol, dose_1d=dose_1d, struct=struct, volume_per=volume)
                        if '%' in str(df.Limit[ind]) or '%' in str(df.Goal[ind]): # we are writing str since nan values throws error
                            df.at[ind, sol_names[p]] = np.round(dose/my_plan.get_prescription()*100, 2)
                        elif 'Gy' in str(df.Limit[ind]) or 'Gy' in str(df.Goal[ind]):
                            df.at[ind, sol_names[p]] = np.round(dose, 2)
        df.round(2)
        for sol_name in sol_names:
            if sol_name not in df:
                df[sol_name] = ''
        df = df[df[sol_names].notna().all(axis=1)]  # remove rows for which plan value is Nan
        df = df.fillna('')
        df['Constraint'] = df['Constraint'].replace({
            'max_dose': 'Max Dose',
            'mean_dose': 'Mean Dose'
        })
        # df.dropna(axis=0, inplace=True)  # remove structures which are not present
        # df.reset_index(drop=True, inplace=True)  # reset the index

        def generate_criteria_bar(limit: float = None, goal: float = None, val: float=None) -> str:
            vals = [v for v in [limit, goal, val] if v is not None]
            min_val = min(vals) * 0.9
            max_val = max(vals) * 1.1
            range_val = max_val - min_val or 1e-5

            def get_pos(x):
                return ((x - min_val) / range_val) * 100

            # Colored background
            colored_regions = ""
            green = "#8DC820"
            orange = "#FC8D0A"
            red = "#FB3D00"

            if limit is not None and goal is not None:
                if limit < goal:
                    # Red → Orange → Green
                    red_pos = get_pos(limit)
                    orange_pos = get_pos(goal)
                    colored_regions = (
                        f'<div style="position:absolute; left:0%; width:{red_pos}%; height:100%; background-color:{red};"></div>'
                        f'<div style="position:absolute; left:{red_pos}%; width:{orange_pos - red_pos}%; height:100%; background-color:{orange};"></div>'
                        f'<div style="position:absolute; left:{orange_pos}%; width:{100 - orange_pos}%; height:100%; background-color:{green};"></div>'
                    )
                else:
                    # Green → Orange → Red
                    green_pos = get_pos(goal)
                    orange_pos = get_pos(limit)
                    colored_regions = (
                        f'<div style="position:absolute; left:0%; width:{green_pos}%; height:100%; background-color:{green};"></div>'
                        f'<div style="position:absolute; left:{green_pos}%; width:{orange_pos - green_pos}%; height:100%; background-color:{orange};"></div>'
                        f'<div style="position:absolute; left:{orange_pos}%; width:{100 - orange_pos}%; height:100%; background-color:{red};"></div>'
                    )
            elif goal is not None:
                goal_pos = get_pos(goal)
                colored_regions = (
                    f'<div style="position:absolute; left:0%; width:{goal_pos}%; height:100%; background-color:{green};"></div>'
                    f'<div style="position:absolute; left:{goal_pos}%; width:{100 - goal_pos}%; height:100%; background-color:{orange};"></div>'
                )
            elif limit is not None:
                limit_pos = get_pos(limit)
                colored_regions = (
                    f'<div style="position:absolute; left:0%; width:{limit_pos}%; height:100%; background-color:{green};"></div>'
                    f'<div style="position:absolute; left:{limit_pos}%; width:{100 - limit_pos}%; height:100%; background-color:{red};"></div>'
                )
            else:
                colored_regions = '<div style="position:absolute; left:0%; width:100%; height:100%; background-color:lightgray;"></div>'

            # Pointer + label above
            val_pos = get_pos(val)
            pointer = (
                f'<div style="position:absolute; left:{val_pos}%; top:-16px; transform:translateX(-50%); font-size:13px;">{val:.2f}</div>'
                f'<div style="position:absolute; left:{val_pos}%; top:0px; transform:translateX(-50%);">'
                f'<div style="width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-top:10px solid black;"></div>'
                f'</div>'
            )

            bar = (
                f'<div style="position:relative; width:250px; height:30px; border:1px solid #ccc; border-radius:4px; margin:4px 0;">'
                f'{colored_regions}{pointer}</div>'
            )
            return bar

        # Convert all relevant columns (plan values) to 'object' before the loop starts
        df_no_format = df.copy()
        for sol_name in sol_names:
            df[sol_name] = df[sol_name].astype(object)

        for ind in df.index:
            for sol_name in sol_names:
                try:
                    limit = float(re.findall(r"[-+]?(?:\d*\.*\d+)", str(df.loc[ind, 'Limit']))[0]) if df.loc[
                        ind, 'Limit'] else None
                    goal = float(re.findall(r"[-+]?(?:\d*\.*\d+)", str(df.loc[ind, 'Goal']))[0]) if df.loc[
                        ind, 'Goal'] else None
                    plan_val = float(df.loc[ind, sol_name])
                    df.at[ind, sol_name] = generate_criteria_bar(limit, goal, plan_val)
                except Exception as e:
                    print(f"Skipping row {ind} due to error: {e}")

        sol_names.append('Limit')
        sol_names.append('Goal')
        # df.style.set_properties(**{'text-align': 'left'})

        # styled_df = df.style.apply(color_plan_value, subset=sol_names, axis=1)  # apply
        styled_df = (
            df.style
            .set_properties(**{
                'text-align': 'left',
                'padding-top': '10px'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'left')]}  # aligns headers
            ])
            .format(precision=2)
        )
        # color to dataframe using df.style method
        if return_df:

            return styled_df
        if in_browser:
            if path is None:
                path = os.getcwd()
            html_file_path = os.path.join(path, html_file_name)
            # Add extra styles only for browser HTML
            styled_df_for_html = styled_df.set_table_styles([
                {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '1px solid black')]},
                {'selector': 'th, td', 'props': [('border', '1px solid black'), ('text-align', 'left')]}
            ], overwrite=True)
            with open(html_file_path, 'w') as f:
                f.write(styled_df_for_html.to_html(escape=False))
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
                print(tabulate(df_no_format, headers='keys', tablefmt='psql'))  # print in console using tabulate

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
