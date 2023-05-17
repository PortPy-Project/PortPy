"""

This example shows creating and modification of structures using portpy_photon

1- Loading portpy data
2- Perform boolean operation on structures and visualize it
3- Create margin around structure and visualize it

"""

import portpy.photon as pp


def ex_3_structure_operations():
    """
    1- Loading portpy data

    """
    # specify the patient data location.
    data_dir = r'../data'
    # display the existing patients in console or browser.
    data = pp.DataExplorer(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Phantom_Patient_1'

    # Load ct, structure and beams as an object
    ct = pp.CT(data)
    structs = pp.Structures(data)

    """
    2- Perform boolean operation on structures and visualize it
    
    """
    # boolean or create margin_mm around structures
    # create a structure PTV-GTV using boolean operation and save it as a new structure called 'PTV_GTV'
    structs.subtract(struct_1_name='PTV', struct_2_name='GTV', new_struct_name='PTV_GTV')
    pp.Visualization.plot_2d_slice(ct=ct, structs=structs, slice_num=60, struct_names=['PTV_GTV'])

    """
    3- Create margin around structure and visualize it
    
    """
    # Modify PTV by creating margin around PTV
    structs.expand(struct_name='PTV', margin_mm=5, new_struct_name='PTV')
    pp.Visualization.plot_2d_slice(ct=ct, structs=structs, slice_num=60, struct_names=['PTV'])


if __name__ == "__main__":
    ex_3_structure_operations()
