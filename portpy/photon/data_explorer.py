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


class DataExplorer:
    """
    A class for exploring the benchmark data available in PortPy

    - **Attributes** ::

        :param data_dir: local directory containing data
        :param patient_id: patient id of the patient

    - **Methods** ::
        :display_patient_metadata()
            display metadata for the patient
        :display_list_of_patients()
            display the patient list in portpy database


    """
    def __init__(self, data_dir: str = None, patient_id: str = None):
        if data_dir is None:
            data_dir = os.path.join('..', 'data')
        self.data_dir = data_dir
        if patient_id is not None:
            self.patient_id = patient_id

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
        if not os.path.exists(pat_dir):  # check if valid directory
            raise Exception("Invalid data directory. Please input valid directory")

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

        display_dict = {}  # we add all the relevant information from meta_data to this dictionary
        data_dir = self.data_dir
        if not os.path.exists(data_dir):  # check if valid directory
            raise Exception("Invalid data directory. Please input valid directory")
        pat_ids = os.listdir(data_dir)
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
            elif key == 'beamlets':  # added this part to check if there are beamlets since beamlets are list of dictionary
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