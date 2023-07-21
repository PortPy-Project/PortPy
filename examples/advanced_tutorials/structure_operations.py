"""

PortPy includes predefined delineated structures. However, users have the option to create
new structures by expanding (*expand()*) or shrinking (*shrink()*) the existing ones, or
by combining them using Boolean operations (*union(), intersect(), subtract()*).
This tutorial will guide you through the following processes:

1. Loading a new patient dataset
2. Performing Boolean operations on the existing structures
3. Expanding a given structure

"""

import portpy.photon as pp


def structure_operations():
    """
    1-  Loading a new patient dataset

    """
    # specify the patient data location.
    data_dir = r'../../data'
    # display the existing patients in console or browser.
    data = pp.DataExplorer(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Phantom_Patient_1'

    # Load ct, structure and beams as an object
    ct = pp.CT(data)
    structs = pp.Structures(data)

    """
    2- Performing Boolean operations on the existing structures
    **Note:** If you aim to generate a new structure while preserving the existing ones, 
    ensure that the new structure is given a unique name that doesn't overlap with any existing patient structure names. 
    If an existing name is used, the new structure will overwrite the old one
    
    """

    # boolean or create margin_mm around structures
    # create a structure PTV-GTV using boolean operation and save it as a new structure called 'PTV_GTV'
    structs.subtract(struct_1_name='PTV', struct_2_name='GTV', new_struct_name='PTV_GTV')
    pp.Visualization.plot_2d_slice(ct=ct, structs=structs, slice_num=60, struct_names=['PTV_GTV'])

    """
    3- Expanding a given structure
    
    """
    # Expand the PTV stucture. We use the same name for the new structure so it would replace the old PTV
    structs.expand(struct_name='PTV', margin_mm=5, new_struct_name='PTV')
    pp.Visualization.plot_2d_slice(ct=ct, structs=structs, slice_num=60, struct_names=['PTV'])


if __name__ == "__main__":
    structure_operations()
