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

"""MedNeXt dose-regression wrapper used by the ``mednext`` generator.

This is the GDP-HMM Top-1 style baseline (Abdulkadir & Neylon). Moved out of
``networks3d.py`` verbatim; behaviour is unchanged. See ``mednext_attention.py``
(Top-2) and ``mednext_duo.py`` (Top-3) for the other challenge solutions.
"""
import warnings

import torch.nn as nn


class MedNeXtDose(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        model_id: str = 'B',
        kernel_size: int = 3,
        deep_supervision: bool = False,

    ):
        super().__init__()
        create_mednext_v1 = None
        try:
            from nnunet_mednext import create_mednext_v1
        except ImportError:
            warnings.warn(
                "MedNeXt is not installed. Install it with "
                "`pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
            )

        if create_mednext_v1 is None:
            raise ImportError(
                "Requested MedNeXtDose but MedNeXt is not installed. "
                "Install it with "
                "`pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
            )
        from typing import Any, cast
        create_mednext_v1 = cast(Any, create_mednext_v1)
        # Keep open kbp style loading path
        self.model = create_mednext_v1(
            num_input_channels=in_channels,
            num_classes=out_channels,
            model_id=model_id,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
        )

        # self.out_act = nn.ReLU(inplace=True)

    def forward(self, x, site=None):
        # `site` is accepted and ignored: only multi-site models use it.
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y
