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

import os
import numpy as np
from scipy.sparse import csr_matrix
import h5py
import json
from natsort import natsorted
from pathlib import Path
import pandas as pd
from tabulate import tabulate
import webbrowser
import posixpath  # Added for HF path construction
from typing import List, Dict, Set, Optional, Tuple, Union

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    # from datasets import load_dataset, DatasetDict # Not strictly needed for this implementation
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

class DataExplorer:
    """
    A class for exploring the benchmark data available in PortPy

    - **Attributes** ::

        :param data_dir: local directory containing data
        :param patient_id: patient id of the patient
        :param hf_repo_id: Hugging Face repository ID (e.g., "PortPy-Project/PortPy_Dataset")
        :param local_download_dir: Alias for data_dir when using Hugging Face, for clarity.
                                   If provided, overrides data_dir for HF downloads.

    - **Methods** ::
        :display_patient_metadata()
            display metadata for the patient
        :display_list_of_patients()
            display the patient list in portpy database


    """
    def __init__(self, data_dir: str = None, patient_id: str = None, hf_repo_id: str = None, hf_token: str = None, local_download_dir: str = None):
        if local_download_dir:
            self.data_dir = local_download_dir
        elif data_dir:
            self.data_dir = data_dir
        elif hf_repo_id:  # If HF repo is given but no specific local dir, use a default
            self.data_dir = os.path.join('..', 'hugging_face_data')
        else:
            self.data_dir = os.path.join('..', 'data')
        if patient_id is not None:
            self.patient_id = patient_id

        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token

        # Ensure the download directory exists if we're using Hugging Face
        if self.hf_repo_id:
            os.makedirs(self.data_dir, exist_ok=True)
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError(
                    "huggingface_hub library is not installed. Please install it to use Hugging Face datasets: pip install huggingface_hub")


    def display_patient_metadata(self, in_browser: bool = False, return_beams_df: bool = False,
                                 return_structs_df: bool = False):
        """Displays the patient information in console or browser. If in_browser is true,
        it creates a temporary html file and lnunches your browser

        :param in_browser: visualize in pretty way in browser. default to False. If false, plot table in console
        :param return_beams_df: return dataframe containing beams metadata
        :param return_structs_df: return dataframe containing structures metadata
        :raises invalid directory error: raises an exception if invalid data directory

        :Example:
        >>> DataExplorer.display_patient_metadata(in_browser=False, return_beams_df=False, return_structs_df=False)
        """

        pat_dir = os.path.join(self.data_dir, self.patient_id)
        if not os.path.exists(pat_dir):
            # Check if it's nested under a 'data' subdirectory, common with hf_hub_download's behavior
            # when filename in repo has 'data/...' prefix.
            potential_pat_dir = os.path.join(self.data_dir, "data", self.patient_id)
            if os.path.exists(potential_pat_dir):
                self.data_dir = os.path.join(self.data_dir, "data") #update data directory
                pat_dir = potential_pat_dir
            else:
                raise Exception(
                    f"Invalid data directory for patient {self.patient_id}. Checked: {pat_dir} and {potential_pat_dir}")

        meta_data = self.load_metadata(pat_dir)
        # if show_beams:
        # check if full and/or sparse influence matrices are provided.
        # Sparse matrix is just a truncated version of the full matrix (zeroing out small elements)
        # often used in the optimization for computational efficiency
        beams = meta_data['beams']  # get beams metadata
        beams_df = pd.DataFrame.from_dict(beams)  # using Pandas data struct_name
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

        if return_beams_df and return_structs_df:
            return beams_df, struct_df
        elif return_beams_df:
            return beams_df
        elif return_structs_df:
            return struct_df
        # Write the results in a temporary html file in the current directory and launch a browser to display
        if in_browser:
            style_file = os.path.join('../..', 'df_style.css')
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
            # create a temporary html file to store data for visualization in browser
            with open('temp.html', 'w') as f:
                f.write(html_string.format(table_1=beams_df.to_html(index=False, header=True, classes='mystyle'),
                                           table_2=struct_df.to_html(index=False, header=True, classes='mystyle'),
                                           style=style_file))
            webbrowser.open('file://' + os.path.realpath('temp.html'))
        else:
            if DataExplorer.is_notebook():
                from IPython.display import display
                print('Beams table..')
                with pd.option_context('display.max_rows', None,
                                       'display.max_columns', None,
                                       'display.precision', 3,
                                       'display.colheader_justify', 'center',
                                       ):
                    beams_df = beams_df.style.set_properties(**{'text-align': 'center'})
                    display(beams_df)
                print('Structure table..')
                with pd.option_context('display.max_rows', None,
                                       'display.max_columns', None,
                                       'display.precision', 3,
                                       'display.colheader_justify', 'center',
                                       ):
                    struct_df = struct_df.style.set_properties(**{'text-align': 'center'})
                    display(struct_df)
            else:
                print('Beams table..')
                print(tabulate(beams_df, headers='keys', tablefmt='psql'))  # print the table in console using tabulate
                print('\n\nStructures table..')
                print(tabulate(struct_df, headers='keys', tablefmt='psql'))

    def display_list_of_patients(self, in_browser: bool = False, return_df: bool = False):
        """
        Displays the list of patients included in data_dir folder in console (by default) or browser (if in_browser=true).
        If in_browser is true, it creates a temporary html file and lunches your browser

        :param data_dir: folder including patient data.
            If it is None, then it assumes the data is in the current directory under sub-folder named "data"
        :param in_browser: If true, it first saves the data in a temporary html file and then lunches the browser to visualize the data (this provides better visualization). Default to False. If false, plot table in console
        :param return_df: return dataframe instead of visualization
        :raises invalid directory error: raises an exception if invalid data directory.

        :return patient data in Panda table

        :Example:
        >>> DataExplorer.display_list_of_patients(data_dir='path/to/data', in_browser=True)

        """
        if self.hf_repo_id is not None:
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("huggingface_hub is required to list patients from Hugging Face.")

            print(f"Fetching patient list from Hugging Face repository: {self.hf_repo_id}")
            # Try to get basic info from data_info.jsonl first
            df = self._get_hf_patient_list_from_data_info()

        else:
            display_dict = {}  # we add all the relevant information from meta_data to this dictionary
            data_dir = self.data_dir
            if not os.path.exists(data_dir):  # check if valid directory
                raise Exception("Invalid data directory. Please input valid directory")
            pat_ids = natsorted(os.listdir(data_dir))
            for i, pat_id in enumerate(pat_ids):  # Loop through patients in path
                if "Patient" in pat_id:  # ignore irrelevant folders
                    display_dict.setdefault('patient_id', []).append(pat_id)
                    meta_data = self.load_metadata(os.path.join(data_dir, pat_id))  # load metadata for the patients
                    # set the keys and append to display dict
                    display_dict.setdefault('disease_site', []).append(pat_id.split('_')[0])
                    ind = meta_data['structures']['name'].index('PTV')
                    display_dict.setdefault('ptv_vol_cc', []).append(meta_data['structures']['volume_cc'][ind])
                    display_dict.setdefault('num_beams', []).append(len(meta_data['beams']['ID']))
                    # check if all the iso centers are same for beams
                    # res = all(
                    #     ele == meta_data['beams']['iso_center'][0] for ele in meta_data['beams']['iso_center'])
                    # if res:
                    #     display_dict.setdefault('iso_center_shift ', []).append('No')
                    # else:
                    #     display_dict.setdefault('iso_center_shift ', []).append('Yes')
            df = pd.DataFrame.from_dict(display_dict)  # convert dictionary to dataframe
        if return_df:
            return df
        if in_browser:
            style_file = os.path.join('..', 'df_style.css')  # get style file path
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
            # create a temporary html file to store data for visualization in browser
            with open('temp.html', 'w') as f:
                f.write(
                    html_string.format(table=df.to_html(index=False, header=True, classes='mystyle'), style=style_file))
            webbrowser.open('file://' + os.path.realpath('temp.html'))
        else:
            if DataExplorer.is_notebook():
                from IPython.display import display
                with pd.option_context('display.max_rows', None,
                                       'display.max_columns', None,
                                       'display.precision', 3,
                                       ):

                    display(df)
            else:
                print(tabulate(df, headers='keys', tablefmt='psql'))  # print in console using tabulate

    def load_metadata(self, pat_dir: str = None) -> dict:
        """Loads metadata of a patient located in path and returns the metadata as a dictionary

        The data are loaded from the following .Json files:
        1- StructureSet_MetaData.json
            including data about the structures (e.g., PTV, Kidney, Lung)
        2- OptimizationVoxels_MetaData.json
            including patient voxel data (3D cubic voxels of patient body)
        3- CT_MetaData.json
            including patient CT scan data (e.g., size, resolution, ct hounsfield units)
        4- PlannerBeams.json
            including the indices of the beams_dict selected by an expert planner based on the geometry/shape/location of tumor/healthy-tissues
        5- ClinicalCriteria_MetaData.json
            including clinically relevant metrics used to evaluate a plan (e.g., Kidney mean dose_1d <= 20Gy, Cord max dose_1d <= 10 Gy)
        6- Beams.json
            including beam information (e.g., gantry angle, collimator angle)

        :param pat_dir: full path of patient folder
        :return: a dictionary including all metadata
        """
        if pat_dir is None:
            pat_dir = os.path.join(self.data_dir, self.patient_id)
        meta_data = dict()  # initialization

        # read information regarding the structures
        fname = os.path.join(pat_dir, 'StructureSet_MetaData.json')
        # Opening JSON file
        json_data = self.load_json(fname)
        meta_data['structures'] = DataExplorer.list_to_dict(json_data)

        # read information regarding the voxels
        fname = os.path.join(pat_dir, 'OptimizationVoxels_MetaData.json')
        # Opening JSON file
        json_data = self.load_json(fname)
        meta_data['opt_voxels'] = DataExplorer.list_to_dict(json_data)

        # read information regarding the CT voxels
        fname = os.path.join(pat_dir, 'CT_MetaData.json')
        if os.path.isfile(fname):
            # Opening JSON file
            json_data = self.load_json(fname)
            meta_data['ct'] = DataExplorer.list_to_dict(json_data)

        # read information regarding beam angles selected by an expert planner
        fname = os.path.join(pat_dir, 'PlannerBeams.json')
        if os.path.isfile(fname):
            # Opening JSON file
            json_data = self.load_json(fname)
            meta_data['planner_beam_ids'] = DataExplorer.list_to_dict(json_data)

        # read information regarding the clinical evaluation metrics
        # fname = os.path.join(pat_dir, 'ClinicalCriteria_MetaData.json')
        # json_data = self.load_json(fname)
        # meta_data['clinical_criteria'] = DataExplorer.list_to_dict(json_data)

        # read information regarding the beams_dict
        beamFolder = os.path.join(pat_dir, 'Beams')
        beamsJson = [pos_json for pos_json in os.listdir(beamFolder) if pos_json.endswith('.json')]

        beamsJson = natsorted(beamsJson)
        meta_data['beams'] = dict()
        # the information for each beam is stored in an individual .json file, so we loop through them
        for i in range(len(beamsJson)):
            fname = os.path.join(beamFolder, beamsJson[i])
            json_data = self.load_json(fname)
            for key in json_data:
                meta_data['beams'].setdefault(key, []).append(json_data[key])
                # dataMeta['beamsMetaData'][key].append(json_data[key])

        meta_data['patient_folder_path'] = pat_dir
        return meta_data

    @staticmethod
    def load_config_clinical_criteria(protocol_name: str, protocol_type: str = 'Default') -> dict:
        """
        Returns json file as dictionary
        :param protocol_name: Clinical protocol name
        :param protocol_type: Default.
        :return: dictionary containing clinical criteria
        """
        # load clinical criteria config metadata
        fname = os.path.join(Path(__file__).parents[1], 'config_files', 'clinical_criteria',
                             protocol_type, protocol_name + '.json')
        # fname = os.path.join('..', 'config_files', 'planner_plan', patient_id, 'planner_plan.json')
        # Opening JSON file
        json_data = DataExplorer.load_json(fname)
        return json_data

    @staticmethod
    def load_config_opt_params(protocol_name: str) -> dict:
        """

        Returns json file as dictionary
        :param protocol_name: Clinical protocol name
        :return: dictionary containing optimization params

        """
        # load opt params config metadata
        fname = os.path.join(Path(__file__).parents[1], 'config_files', 'optimization_params',
                             'optimization_params_' + protocol_name + '.json')
        # fname = os.path.join('..', 'config_files', 'planner_plan', patient_id, 'planner_plan.json')
        # Opening JSON file
        json_data = DataExplorer.load_json(fname)
        return json_data

    @staticmethod
    def load_config_tcia_patients() -> dict:
        """
        Returns TCIA patients metadata

        """
        # load clinical criteria config metadata
        fname = os.path.join(Path(__file__).parents[1], 'config_files', 'tcia_patients', 'tcia_patients_metadata.json')
        # fname = os.path.join('..', 'config_files', 'planner_plan', patient_id, 'planner_plan.json')
        # Opening JSON file
        json_data = DataExplorer.load_json(fname)
        return json_data

    def get_tcia_metadata(self, patient_id: str = None):
        """
        Returns tcia patient metadata for the given PortPy patient id

        """
        if patient_id is None:
            patient_id = self.patient_id
        # load clinical criteria config metadata
        fname = os.path.join(Path(__file__).parents[1], 'config_files', 'tcia_patients', 'tcia_patients_metadata.json')
        # fname = os.path.join('..', 'config_files', 'planner_plan', patient_id, 'planner_plan.json')
        # Opening JSON file
        json_data = DataExplorer.load_json(fname)
        # Adjust smoothness weight in the objective function to have appropriate MU
        patient_found = False
        for i in range(len(json_data['tcia_patients'])):
            if json_data['tcia_patients'][i]['portpy_patient_id'] == patient_id:
                print(json_data['tcia_patients'][i])
                patient_found = True
        if not patient_found:
            print('Invalid patient id')

    @staticmethod
    def load_json(file_name):
        f = open(file_name)
        json_data = json.load(f)
        f.close()
        return json_data

    def load_data(self, meta_data: dict, load_inf_matrix_full: bool = False) -> dict:
        """
        Takes meta_data and the location of the data as inputs and returns the full data.
        The meta_data only includes light-weight data from the .json files (e.g., beam IDs, angles, struct_name names,..).
        Large numeric data (e.g., influence matrix, voxel coordinates) are stored in .h5 files.


        :param load_inf_matrix_full: whether to load full influence matrix from the data
        :param meta_data: meta_data containing light weight data from json file
        :param pat_dir: patient folder directory containing all the data
        e.g. if options['loadInfluenceMatrixFull']=True, it will load full influence matrix
        :return: a dict of data
        """
        if not load_inf_matrix_full:
            if 'influenceMatrixFull_File' in meta_data:
                meta_data['influenceMatrixFull_File'] = [None] * len(
                    meta_data['influenceMatrixFull_File'])
        elif load_inf_matrix_full:
            if 'influenceMatrixSparse_File' in meta_data:
                meta_data['influenceMatrixSparse_File'] = [None] * len(
                    meta_data['influenceMatrixSparse_File'])
        meta_data = self.load_file(meta_data=meta_data)  # recursive function to load data from .h5 files
        return meta_data

    def load_file(self, meta_data: dict):
        """
        This recursive function loads the data from .h5 files and merge them with the meta_data and returns a dictionary
        including all the data (meta_data+actual numeric data)
        :param meta_data: meta_data containing leight weight data from json file
        :param pat_dir: patient folder directory
        :return:
        """
        for key in meta_data.copy():
            item = meta_data[key]
            if type(item) is dict:
                meta_data[key] = self.load_file(item)
            elif key == 'beamlets' or key == 'spots':  # added this part to check if there are beamlets since beamlets are list of dictionary
                if type(item[0]) is dict:
                    for ls in range(len(item)):
                        self.load_file(item[ls])
                        # meta_data[key] = ls_data
            elif key.endswith('_File'):
                success = 1
                for i in range(np.size(meta_data[key])):
                    dataFolder = os.path.join(self.data_dir, self.patient_id)
                    if meta_data[key][i] is not None:
                        if meta_data[key][i].startswith('Beam_'):
                            dataFolder = os.path.join(dataFolder, 'Beams')
                        if type(meta_data[key]) is not list:
                            if meta_data[key].startswith('Beam_'):  # added this for beamlets
                                dataFolder = os.path.join(dataFolder, 'Beams')
                            file_tag = meta_data[key].split('.h5')
                        else:
                            file_tag = meta_data[key][i].split('.h5')
                        filename = os.path.join(dataFolder, file_tag[0] + '.h5')
                        with h5py.File(filename, "r") as f:
                            if file_tag[1] in f:
                                if key[0:-5] == 'optimizationVoxIndices':
                                    vox = f[file_tag[1]][:].ravel()
                                    meta_data.setdefault(key[0:-5], []).append(vox.astype(int))
                                elif key[0:-5] == 'BEV_2d_structure_mask':
                                    orgs = f[file_tag[1]].keys()
                                    organ_mask_dict = dict()
                                    for j in orgs:
                                        organ_mask_dict[j] = f[file_tag[1]][j][:]
                                    #                                     organ_mask_dict['Mask'].append(f[file_tag[1]][j][:].T)
                                    meta_data.setdefault(key[0:-5], []).append(organ_mask_dict)
                                elif key[0:-5] == 'BEV_structure_contour_points':
                                    orgs = f[file_tag[1]].keys()
                                    organ_mask_dict = dict()
                                    for j in orgs:
                                        segments = f[file_tag[1]][j].keys()
                                        for seg in segments:
                                            organ_mask_dict.setdefault(j, []).append(f[file_tag[1]][j][seg][:])
                                            # organ_mask_dict[j] = f[file_tag[1]][j][seg][:].T
                                    #                                     organ_mask_dict['Mask'].append(f[file_tag[1]][j][:].T)
                                    meta_data.setdefault(key[0:-5], []).append(organ_mask_dict)
                                #                                 meta_data.setdefault(key[0:-5], []).append(f[file_tag[1]][j][:].T)
                                else:
                                    meta_data.setdefault(key[0:-5], []).append(f[file_tag[1]][:])
                                if key[0:-5] == 'influenceMatrixSparse':
                                    infMatrixSparseForBeam = meta_data[key[0:-5]][i]
                                    meta_data[key[0:-5]][i] = csr_matrix(
                                        (infMatrixSparseForBeam[:, 2], (infMatrixSparseForBeam[:, 0].astype(int),
                                                                        infMatrixSparseForBeam[:, 1].astype(int))))
                            else:
                                print('Problem reading data: {}'.format(meta_data[key][i]))
                                success = 0
                if success:
                    del meta_data[key]

        return meta_data

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

    def _get_hf_patient_list_from_data_info(self, quiet: bool = False) -> pd.DataFrame:
        """
        Fetches patient IDs and disease sites from data_info.jsonl in the Hugging Face repo.
        """
        if not self.hf_repo_id:
            if not quiet:
                print("Hugging Face repository ID (hf_repo_id) not set.")
            return pd.DataFrame()
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub is required for this feature.")

        try:
            data_info_file_path = "data_info.jsonl"
            if not quiet:
                print(f"Fetching patient list from {self.hf_repo_id}/{data_info_file_path}...")
            local_file = hf_hub_download(
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                filename=data_info_file_path,
                local_dir=os.path.join(self.data_dir, ".hf_cache")  # Temp cache for this file
                # use_auth_token=self.hf_token
            )
            with open(local_file) as f:
                data_info = [json.loads(line) for line in f]

            patient_ids = [pat.get('patient_id') for pat in data_info if pat.get('patient_id')]
            if not patient_ids:
                if not quiet:
                    print("No patient_id found in data_info.jsonl")
                return pd.DataFrame()

            df = pd.DataFrame(patient_ids, columns=["patient_id"])
            # Attempt to extract disease site common pattern
            df["disease_site"] = df["patient_id"].apply(
                lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else "Unknown")

            # Estimate num_beams from beam_metadata_paths
            df["num_beams"] = [
                len(pat.get("beam_metadata_paths", [])) if isinstance(pat.get("beam_metadata_paths"), list) else None
                for pat in data_info
            ]
            return df
        except Exception as e:
            if not quiet:
                print(f"Error fetching or parsing data_info.jsonl from Hugging Face: {e}")
            return pd.DataFrame()

    def _load_structure_metadata(self, patient_id: str, temp_download_dir: Optional[str] = None,
                                 local_data_dir: Optional[str] = None) -> list:
        """Loads structure metadata from local disk if available; otherwise from Hugging Face."""
        filename = "StructureSet_MetaData.json"
        if local_data_dir:
            local_file = os.path.join(local_data_dir, patient_id, filename)
            if os.path.exists(local_file):
                try:
                    with open(local_file) as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load local structure metadata for {patient_id}: {e}")

        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub is required for Hugging Face access.")

        try:
            hf_path = posixpath.join("data", patient_id, filename)
            local_file = hf_hub_download(
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                filename=hf_path,
                local_dir=temp_download_dir,
                use_auth_token=self.hf_token
            )
            with open(local_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load structure metadata for {patient_id} from HF: {e}")
            return []

    def _load_beam_metadata(self, patient_id: str, temp_download_dir: Optional[str] = None,
                            local_data_dir: Optional[str] = None) -> list:
        """Loads beam metadata from local disk if available; otherwise from Hugging Face."""
        beam_meta = []

        if local_data_dir:
            local_beam_dir = os.path.join(local_data_dir, patient_id, "Beams")
            if os.path.isdir(local_beam_dir):
                try:
                    for fname in os.listdir(local_beam_dir):
                        if fname.endswith("_MetaData.json"):
                            with open(os.path.join(local_beam_dir, fname)) as f:
                                beam_meta.append(json.load(f))
                    if beam_meta:
                        return beam_meta
                except Exception as e:
                    print(f"Warning: Could not load local beam metadata for {patient_id}: {e}")

        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub is required for Hugging Face access.")

        try:
            repo_files = list_repo_files(repo_id=self.hf_repo_id, repo_type="dataset", token=self.hf_token)
            beam_meta_paths = [
                f for f in repo_files
                if f.startswith(f"data/{patient_id}/Beams/Beam_") and f.endswith("_MetaData.json")
            ]
            for path in beam_meta_paths:
                local_file = hf_hub_download(
                    self.hf_repo_id,
                    repo_type="dataset",
                    filename=path,
                    local_dir=temp_download_dir,
                    use_auth_token=self.hf_token
                )
                with open(local_file) as f:
                    beam_meta.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load beam metadata for {patient_id} from HF: {e}")

        return beam_meta

    def _load_planner_beams(self, patient_id: str, temp_download_dir: Optional[str] = None,
                            local_data_dir: Optional[str] = None) -> list:
        """Loads planner beam IDs from local disk if available; otherwise from Hugging Face."""
        filename = "PlannerBeams.json"
        if local_data_dir:
            local_file = os.path.join(local_data_dir, patient_id, filename)
            if os.path.exists(local_file):
                try:
                    with open(local_file) as f:
                        return json.load(f).get("IDs", [])
                except Exception as e:
                    print(f"Warning: Could not load local planner beams for {patient_id}: {e}")

        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub is required for Hugging Face access.")

        try:
            hf_path = posixpath.join("data", patient_id, filename)
            local_file = hf_hub_download(
                self.hf_repo_id,
                repo_type="dataset",
                filename=hf_path,
                local_dir=temp_download_dir,
                use_auth_token=self.hf_token
            )
            with open(local_file) as f:
                return json.load(f).get("IDs", [])
        except Exception as e:
            print(f"Warning: Could not load planner beams for {patient_id} from HF: {e}")
            return []

    def _get_hf_patient_summary(self, patient_id: str, temp_download_dir: str) -> dict:
        """Gets PTV volume and beam count for a patient by downloading minimal HF metadata."""
        structs = self._load_structure_metadata(patient_id, temp_download_dir=temp_download_dir)
        beams = self._load_beam_metadata(patient_id, temp_download_dir=temp_download_dir)
        planner_beam_ids = self._load_planner_beams(patient_id, temp_download_dir=temp_download_dir)

        ptv_vol = None
        if structs:
            for s in structs:
                if "PTV" in s.get("name", "").upper():
                    ptv_vol = s.get("volume_cc")
                    break
        return {
            "ptv_vol_cc": ptv_vol,
            "num_beams": len(beams),
            "beams_metadata": beams,  # For detailed filtering
            "structures_metadata": structs,  # For detailed filtering
            "planner_beam_ids": planner_beam_ids
        }

    def filter_and_download_hf_dataset(self,
                                       disease_site_filter: str = None,
                                       patient_ids: List[str] = None,
                                       min_ptv_volume_cc: float = None,
                                       gantry_angles_filter: List[float] = None,  # List of floats
                                       collimator_angles_filter: List[float] = None,  # List of floats
                                       couch_angles_filter: List[float] = None,  # List of floats
                                       energies_filter: List[str] = None,  # List of strings
                                       beam_ids_to_download: Union[List[Union[int, str]], np.ndarray] = None,
                                       use_planner_beams_only: bool = True,
                                       max_patients_to_download: int = None):
        """
        Filters patients from the configured Hugging Face dataset based on criteria and downloads their data.

        :param disease_site_filter: Disease site to filter by (e.g., "Lung").
        :param patient_ids: (Optional) List of patient ids to be downloaded
        :param min_ptv_volume_cc: Minimum PTV volume in cc.
        :param gantry_angles_filter: Comma-separated string of desired gantry angles.
        :param collimator_angles_filter: Comma-separated string of desired collimator angles.
        :param couch_angles_filter: Comma-separated string of desired couch angles.
        :param energies_filter: Comma-separated string of desired beam energies (e.g., "6X", "10X").
        :param beam_ids_to_download: Explicit list/array of beam IDs to download (overrides all beam filtering).
        :param use_planner_beams_only: If True, only consider beams selected by the planner.
                                       If False, considers all available beams matching other criteria.
        :param max_patients_to_download: Maximum number of matched patients to download.
        :return: List of local paths to the downloaded patient folders.
        """
        if not self.hf_repo_id:
            print("Hugging Face repository ID (hf_repo_id) not set.")
            return []
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub is required for this feature.")

        print(f"Starting filtering and download from {self.hf_repo_id}...")

        # Determine output directory
        print(f"Data will be downloaded to: {self.data_dir}")

        # Temporary directory for downloading metadata for filtering
        # This avoids cluttering the main data_dir with metadata from non-matching patients
        temp_filter_meta_dir = os.path.join(self.data_dir, ".hf_cache")
        os.makedirs(temp_filter_meta_dir, exist_ok=True)

        # 1. Get initial list of patients
        all_patients_df = self._get_hf_patient_list_from_data_info()

        # 2. Filter by disease site (if provided in data_info.jsonl)
        if disease_site_filter and "disease_site" in all_patients_df.columns:
            all_patients_df = all_patients_df[
                all_patients_df["disease_site"].str.contains(disease_site_filter, case=False, na=False)]
        elif disease_site_filter:
            print(
                "Warning: Disease site filtering may be incomplete as 'disease_site' column not in data_info.jsonl summary.")
            # If not in summary, we'd have to download metadata for all patients, which is slow.
            # For now, we proceed and filter later if possible or rely on patient ID naming.

        # Convert filter strings to sets for easier lookup
        gantry_angles_set = set(map(float, gantry_angles_filter)) if gantry_angles_filter else None
        collimator_angles_set = set(map(float, collimator_angles_filter)) if collimator_angles_filter else None
        couch_angles_set = set(map(float, couch_angles_filter)) if couch_angles_filter else None
        energies_set = set(map(str, energies_filter)) if energies_filter else None

        matched_patients_data = []
        processed_patient_ids = all_patients_df["patient_id"].tolist()
        if self.patient_id is not None:
            processed_patient_ids = [self.patient_id]
        else:
            processed_patient_ids = patient_ids


        print(
            f"Found {len(processed_patient_ids)} patients matching initial criteria. Fetching metadata for detailed filtering...")

        for i, patient_id in enumerate(processed_patient_ids):
            # print(f"Processing patient {i + 1}/{len(processed_patient_ids)}: {patient_id}")
            # Fetch detailed metadata for this patient
            summary = self._get_hf_patient_summary(patient_id, temp_filter_meta_dir)
            structures_metadata = summary.get("structures_metadata", [])
            beams_metadata = summary.get("beams_metadata", [])

            # Apply disease site filter again if not done via data_info.jsonl or if it was ambiguous
            if disease_site_filter and not structures_metadata:  # fallback if disease_site not well defined in data_info
                if disease_site_filter.lower() not in patient_id.lower():
                    continue

            # PTV Volume Filter
            if min_ptv_volume_cc is not None:
                current_ptv_vol = summary.get("ptv_vol_cc")
                if current_ptv_vol is None or current_ptv_vol < min_ptv_volume_cc:
                    continue

            planner_ids_set = set(summary["planner_beam_ids"]) if use_planner_beams_only and summary.get(
                "planner_beam_ids") else None

            beam_ids_to_download = self.filter_beams_by_properties(
                beams_metadata=beams_metadata,
                gantry_angles=gantry_angles_set,
                collimator_angles=collimator_angles_set,
                couch_angles=couch_angles_set,
                energies=energies_set,
                planner_beam_ids=planner_ids_set,
                use_planner_beams_only=use_planner_beams_only,
                beam_ids=beam_ids_to_download
            )

            if not beam_ids_to_download:
                continue

            # If we pass all filters for this patient
            matched_patients_data.append({
                "patient_id": patient_id,
                "beam_ids_to_download": beam_ids_to_download,
                "ptv_volume": summary.get("ptv_vol_cc"),
                "num_matching_beams": len(beam_ids_to_download)
            })
            print(f"Patient {patient_id} matched with {len(beam_ids_to_download)} beams.")

            if max_patients_to_download and len(matched_patients_data) >= max_patients_to_download:
                print(f"Reached max_patients_to_download ({max_patients_to_download}).")
                break

        # # Cleanup temp filter cache
        # try:
        #     import shutil
        #     shutil.rmtree(temp_filter_meta_dir)
        # except OSError as e:
        #     print(f"Warning: Could not remove temporary filter cache directory {temp_filter_meta_dir}: {e}")

        # 3. Download data for matched patients
        downloaded_patient_folders = []
        if not matched_patients_data:
            print("No patients matched the specified criteria.")
            return []

        print(f"\nFound {len(matched_patients_data)} patients matching all criteria. Starting download...")
        for patient_info in matched_patients_data:
            pat_id = patient_info["patient_id"]
            beam_ids = patient_info["beam_ids_to_download"]
            patient_download_dir = os.path.join(self.data_dir, 'data', pat_id)  # TODO. Download into subfolder per patient
            # os.makedirs(patient_download_dir, exist_ok=True)

            print(f"Downloading data for patient: {pat_id}...")
            self._download_hf_patient_data(
                patient_id=pat_id,
                beam_ids_to_download=beam_ids,  # Pass only the filtered beam IDs
                local_patient_dir=patient_download_dir,  # Download directly into the patient's final folder
                download_all_beams_if_empty=not bool(beam_ids) and not any(
                    [gantry_angles_set, collimator_angles_set, couch_angles_set, energies_set, use_planner_beams_only])
                # Download all if no beam filters applied
            )
            downloaded_patient_folders.append(patient_download_dir)
            # Update self.data_dir and self.patient_id if only one patient downloaded for immediate exploration
            self.data_dir = os.path.join(self.data_dir, 'data')
            if len(matched_patients_data) == 1:
                self.patient_id = pat_id
                print(f"DataExplorer automatically set to explore downloaded patient: {pat_id} in {self.data_dir}")

        print("\nDownload complete.")
        print("Downloaded patient folders:", downloaded_patient_folders)
        return downloaded_patient_folders

    def filter_beams_by_properties(
            self,
            beams_metadata: Optional[List[Dict]] = None,
            gantry_angles: Optional[Union[List[float], Set[float]]] = None,
            collimator_angles: Optional[Union[List[float], Set[float]]] = None,
            couch_angles: Optional[Union[List[float], Set[float]]] = None,
            energies: Optional[Union[List[float], Set[float]]] = None,
            planner_beam_ids: Optional[Union[List[float], Set[float]]] = None,
            iso_centers: Optional[Union[List, Set]] = None,
            match_iso_to_planner_beams: bool = True,
            use_planner_beams_only: bool = False,
            beam_ids: Optional[Union[List[Union[int, str]], np.ndarray]] = None
    ) -> List[Union[int, str]]:
        """
        Filters beam IDs based on gantry, collimator, couch, energy, isocenter, and optionally planner beams.

        :param beams_metadata: List of beam metadata dictionaries.
        :param gantry_angles: Set of allowed gantry angles.
        :param collimator_angles: Set of allowed collimator angles.
        :param couch_angles: Set of allowed couch angles.
        :param energies: Set of allowed energy strings (e.g., {"6X", "10X"}).
        :param planner_beam_ids: Set of planner beam IDs (optional).
        :param iso_centers: Set of allowed isocenter coordinates as tuples (x_mm, y_mm, z_mm).
        :param match_iso_to_planner_beams: If True, only beams matching the isocenter(s) of planner beams are kept.
        :param use_planner_beams_only: If True, only planner beams are considered in final result.
        :param beam_ids: Optional[Union[List[Union[int, str]], np.ndarray]] = None,
        :return: List of filtered beam IDs.
        """

        if beam_ids is not None:
            if isinstance(beam_ids, np.ndarray):
                beam_ids = beam_ids.tolist()
            return [str(b) for b in beam_ids]

        # Convert list args to sets for consistent lookup
        if isinstance(gantry_angles, list):
            gantry_angles = set(gantry_angles)
        if isinstance(collimator_angles, list):
            collimator_angles = set(collimator_angles)
        if isinstance(couch_angles, list):
            couch_angles = set(couch_angles)
        if isinstance(energies, list):
            energies = set(energies)
        if isinstance(planner_beam_ids, list):
            planner_beam_ids = set(planner_beam_ids)

        seen_fingerprints = set()
        unique_beam_ids = []

        # Determine reference iso centers from planner beams if needed
        if beams_metadata is None:
            beams_metadata = self._load_beam_metadata(self.patient_id, local_data_dir=self.data_dir)
        if planner_beam_ids is None:
            # read information regarding beam angles selected by an expert planner
            fname = os.path.join(self.data_dir,self.patient_id,'PlannerBeams.json')
            if os.path.isfile(fname):
                # Opening JSON file
                json_data = self.load_json(fname)
                planner_beam_ids = DataExplorer.list_to_dict(json_data)
                planner_beam_ids = planner_beam_ids['IDs']
        if match_iso_to_planner_beams and planner_beam_ids:
            planner_iso_centers = {
                tuple(beam["iso_center"].get(k) for k in ("x_mm", "y_mm", "z_mm"))
                for beam in beams_metadata
                if beam["ID"] in planner_beam_ids
            }
        else:
            planner_iso_centers = None

        selected_beams = []

        for beam in beams_metadata:
            if gantry_angles and beam.get("gantry_angle") not in gantry_angles:
                continue
            if collimator_angles and beam.get("collimator_angle") not in collimator_angles:
                continue
            if couch_angles and beam.get("couch_angle") not in couch_angles:
                continue
            if energies and beam.get("energy_MV") not in energies:
                continue

            iso = beam.get("iso_center", {})
            iso_tuple = (iso.get("x_mm"), iso.get("y_mm"), iso.get("z_mm"))

            if iso_centers and iso_tuple not in iso_centers:
                continue
            if planner_iso_centers and iso_tuple not in planner_iso_centers:
                continue
            # Create a fingerprint based on key beam metadata
            fingerprint = (
                beam.get("gantry_angle"),
                beam.get("collimator_angle"),
                beam.get("couch_angle"),
                beam.get("energy_MV"),
                round(beam.get("iso_center", {}).get("x_mm", 0), 2),
                round(beam.get("iso_center", {}).get("y_mm", 0), 2),
                round(beam.get("iso_center", {}).get("z_mm", 0), 2),
            )
            if fingerprint in seen_fingerprints:
                continue  # skip duplicate
            seen_fingerprints.add(fingerprint)

            selected_beams.append(beam)

        selected_ids = {beam["ID"] for beam in selected_beams}

        if use_planner_beams_only:
            if planner_beam_ids:
                selected_ids = selected_ids & set(planner_beam_ids)
            else:
                return []

        return list(selected_ids)

    def _download_hf_patient_data(self, patient_id: str, beam_ids_to_download: list = None,
                                  local_patient_dir: str = None, max_retries: int = 2,
                                  download_all_beams_if_empty: bool = False):
        """
        Downloads files for a specific patient from Hugging Face.
        If beam_ids_to_download is None or empty and download_all_beams_if_empty is True,
        it will try to download all beams for that patient.

        :param patient_id: The ID of the patient.
        :param beam_ids_to_download: Specific list of beam IDs to download.
        :param local_patient_dir: The local directory to save this patient's data.
                                 If None, defaults to self.data_dir / patient_id.
        :param max_retries: Number of retries for downloading each file.
        :param download_all_beams_if_empty: If beam_ids_to_download is empty/None, download all beams.
        """
        if not self.hf_repo_id:
            print("Hugging Face repository ID (hf_repo_id) not set.")
            return
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub is required for this feature.")

        # Files that are always downloaded for a patient (non-beam specific)
        static_files_templates = [
            "CT_Data.h5", "CT_MetaData.json",
            "StructureSet_Data.h5", "StructureSet_MetaData.json",
            "OptimizationVoxels_Data.h5", "OptimizationVoxels_MetaData.json",
            "PlannerBeams.json",
            # Optional DICOM files - check if they exist before attempting download or make this configurable
            # "rt_dose_echo_imrt.dcm", "rt_plan_echo_imrt.dcm"
        ]
        # Add known DICOM files if they are standard part of your dataset
        optional_dicom_files = ["rt_dose_echo_imrt.dcm", "rt_plan_echo_imrt.dcm"]
        all_repo_files = list_repo_files(repo_id=self.hf_repo_id, repo_type="dataset")

        files_to_download = []
        for filename_template in static_files_templates:
            hf_path = posixpath.join("data", patient_id, filename_template)
            if hf_path in all_repo_files:
                files_to_download.append(hf_path)
            else:
                print(f"Info: Static file {hf_path} not found in repository. Skipping.")

        for dicom_file in optional_dicom_files:
            hf_path = posixpath.join("data", patient_id, dicom_file)
            if hf_path in all_repo_files:  # Check if it exists
                files_to_download.append(hf_path)
            else:
                print(f"Info: Optional DICOM file {hf_path} not found. Skipping.")

        # Determine which beams to download
        actual_beam_ids_to_download = []
        if beam_ids_to_download:  # Specific beams requested
            actual_beam_ids_to_download = beam_ids_to_download
        elif download_all_beams_if_empty:  # Download all beams for this patient
            print(f"No specific beams listed, downloading all available beams for {patient_id}...")
            beam_meta_files = [
                f for f in all_repo_files
                if f.startswith(f"data/{patient_id}/Beams/Beam_") and f.endswith("_MetaData.json")
            ]
            for meta_file_path in beam_meta_files:
                # Extract beam ID from filename like "data/Patient_X/Beams/Beam_Y_MetaData.json" -> Y
                try:
                    beam_id_str = meta_file_path.split('/')[-1].replace("Beam_", "").replace("_MetaData.json", "")
                    actual_beam_ids_to_download.append(beam_id_str)  # Assuming beam IDs are strings here
                except Exception as e:
                    print(f"Warning: Could not parse beam ID from {meta_file_path}: {e}")

        if actual_beam_ids_to_download:
            print(f"Identified beams to download for {patient_id}: {actual_beam_ids_to_download}")
            for bid in actual_beam_ids_to_download:
                # Construct paths for beam data and metadata
                # Ensure correct casing and extensions as per your HF repo
                beam_data_fname = f"Beam_{bid}_Data.h5"
                beam_meta_fname = f"Beam_{bid}_MetaData.json"

                beam_data_hf_path = posixpath.join("data", patient_id, "Beams", beam_data_fname)
                beam_meta_hf_path = posixpath.join("data", patient_id, "Beams", beam_meta_fname)

                if beam_data_hf_path in all_repo_files:
                    files_to_download.append(beam_data_hf_path)
                else:
                    print(
                        f"Warning: Beam data file {beam_data_hf_path} not found for patient {patient_id}, beam {bid}. Skipping.")

                if beam_meta_hf_path in all_repo_files:
                    files_to_download.append(beam_meta_hf_path)
                else:
                    print(
                        f"Warning: Beam metadata file {beam_meta_hf_path} not found for patient {patient_id}, beam {bid}. Skipping.")
        elif beam_ids_to_download is not None and not beam_ids_to_download:  # Empty list explicitly passed and not download_all
            print(f"No beams selected for download for patient {patient_id} based on filters.")

        downloaded_file_paths = []
        for hf_path in files_to_download:
            # Determine the correct local path structure
            # hf_path is like "data/Patient_ID/Beams/Beam_1_Data.h5"
            # We want it in local_patient_dir/"Beams/Beam_1_Data.h5" or local_patient_dir/"StructureSet_Data.h5"
            relative_path_from_patient_data_on_hf = posixpath.relpath(hf_path, posixpath.join("data", patient_id))
            # local_dest_path = os.path.join(local_patient_dir, relative_path_from_patient_data_on_hf)
            # os.makedirs(os.path.dirname(local_dest_path), exist_ok=True) # Ensure subdirectories like "Beams" exist

            for attempt in range(max_retries):
                try:
                    # hf_hub_download will create subdirectories specified in `filename` relative to `local_dir`
                    # So, filename should be the path *within* the patient's data folder on HF
                    # and local_dir should be the patient's root download folder.
                    print(f"  Downloading {hf_path} to {local_patient_dir}...")
                    local_file_actually_downloaded_to = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        repo_type="dataset",
                        filename=hf_path,  # Full path on repo
                        local_dir=self.data_dir,  # This should be the root output dir for filtered data

                    )

                    downloaded_file_paths.append(local_file_actually_downloaded_to)
                    break  # Success
                except Exception as e:
                    print(f"  Attempt {attempt + 1}/{max_retries} failed for {hf_path}: {e}")
                    if attempt == max_retries - 1:
                        print(f"  Failed to download {hf_path} after {max_retries} retries.")
        # return downloaded_file_paths # Not currently used, but good for debugging


    @staticmethod
    def list_to_dict(json_data):
        """
        A recursive function which constructs dictionary from list
        :param json_data: data in json or list format
        :return: data in dictionary format
        """

        json_dict = {}
        if type(json_data) is list:
            for i in range(len(json_data)):
                elem = json_data[i]
                if type(elem) is list:
                    json_dict[i] = DataExplorer.list_to_dict(elem)
                else:
                    for key in elem:
                        json_dict.setdefault(key, []).append(elem[key])
        else:
            json_dict = json_data.copy()
        return json_dict