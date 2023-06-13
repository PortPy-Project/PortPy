from .data_explorer import DataExplorer
from typing import List


class CT:
    """
    Class representing CT for the patient

    - **Attributes** ::

        :param ct_dict: dictionary containing ct metadata and data
        :type patient_id: patient id of the patient


    """

    def __init__(self, data: DataExplorer):
        metadata = data.load_metadata()
        ct_dict = data.load_data(meta_data=metadata['ct'])
        self.ct_dict = ct_dict
        self.patient_id = data.patient_id

    def get_ct_res_xyz_mm(self) -> List[float]:
        """

        :return: number of fractions to be delivered
        """
        return self.ct_dict['resolution_xyz_mm']
