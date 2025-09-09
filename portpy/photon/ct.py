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
