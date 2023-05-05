"""

    This example shows creating and modification of structures using portpy_photon
"""

import portpy.photon as pp


def ex_3_structure_operations():
    # Enter patient name
    # ***************** 0) Creating a plan using the original data resolution **************************
    # Create my_plan object for the planner beams.
    data_dir = r'../../data'
    patient_id = 'Lung_Phantom_Patient_1'
    my_plan = pp.Plan(patient_id, data_dir=data_dir)

    # boolean or create margin_mm around structures
    my_plan.structures.union(str_1='PTV', str_2='GTV', str1_or_str2='dummy')
    my_plan.structures.intersect(str_1='PTV', str_2='GTV', str1_and_str2='dummy')
    my_plan.structures.subtract(str_1='PTV', str_2='GTV', str1_sub_str2='dummy')
    my_plan.structures.expand(structure='PTV', margin_mm=5, new_structure='dummy')


if __name__ == "__main__":
    ex_3_structure_operations()
