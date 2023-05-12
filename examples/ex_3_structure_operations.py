"""

    This example shows creating and modification of structures using portpy_photon
"""

import portpy.photon as pp


def ex_3_structure_operations():
    # specify the patient data location.
    data_dir = r'../data'
    # display the existing patients in console or browser.
    data = pp.DataExplorer(data_dir=data_dir)

    # pick a patient from the existing patient list to get detailed info (e.g., beam angles, structures).
    data.patient_id = 'Lung_Phantom_Patient_1'

    # Load ct, structure and beams as an object
    ct = pp.CT(data)
    structs = pp.Structures(data)

    # boolean or create margin_mm around structures
    structs.subtract(struct_1_name='PTV', struct_2_name='GTV', new_struct_name='PTV_GTV')
    pp.Visualization.plot_2d_slice(ct=ct, structs=structs, slice_num=60, struct_names=['PTV', 'PTV_GTV'])

    structs.expand(struct_name='PTV', margin_mm=5, new_struct_name='PTV')
    pp.Visualization.plot_2d_slice(ct=ct, structs=structs, slice_num=60, struct_names=['PTV'])


if __name__ == "__main__":
    ex_3_structure_operations()
