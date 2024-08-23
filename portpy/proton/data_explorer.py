from portpy.photon import DataExplorer
import os
import pandas as pd
from tabulate import tabulate
import webbrowser

class DataExplorer(DataExplorer):
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

        super().__init__(data_dir=data_dir, patient_id=patient_id)


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
        keep_columns = ['ID', 'gantry_angle', 'couch_angle', 'beam_modality', 'energy_MV',
                        'influence_matrix(sparse/full)',
                        'iso_center',
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