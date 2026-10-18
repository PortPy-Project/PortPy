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
"""Attention-augmented MedNeXt for 3D dose prediction (GDP-HMM Top-2 solution).

Reimplementation of the 2nd-place solution of the AAPM 2025 GDP-HMM Grand
Challenge:

    Tianyu Xiong and Jing Cai, "Technical Report: Generalizable Dose Prediction
    for Heterogeneous Multi-Cohort and Multi-Site Radiotherapy Planning
    (GDP-HMM) Grand Challenge", The Hong Kong Polytechnic University, 2025.

Relative to the plain :class:`MedNeXt` dose model (the Top-1 style baseline in
``networks3d.MedNeXtDose``) the method adds exactly three things:

1. a **channel attention** block (squeeze-and-excitation, Hu et al. 2018) and a
   **spatial attention** block (Oktay et al. 2018) on every skip connection, to
   emphasise salient feature channels and regions;
2. a **ReLU output activation** replacing MedNeXt's segmentation-style head,
   because dose prediction is a regression and not a classification task;
3. training with a larger convolution kernel reached through MedNeXt's UpKern
   schedule (3 -> 5 -> 7). The kernel size is a constructor argument here, so
   the schedule is driven by the training script, not by this class.

Note that MedNeXt fuses skips by ADDITION (``x_res_i + x_up_i``), not by
concatenation, so the attention is applied to the encoder feature ``x_res_i``
before it is added to the upsampled decoder feature.

This subclasses :class:`nnunet_mednext.MedNeXt` rather than wrapping it so that
every backbone parameter keeps its original name. That matters: it keeps the
checkpoints loadable by ``nnunet_mednext.upkern_load_weights`` when growing the
kernel from 3 to 5 to 7.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import torch
import torch.nn as nn

try:
    from nnunet_mednext import MedNeXt as _MedNeXt
except ImportError:  # pragma: no cover - optional dependency
    _MedNeXt = None
    warnings.warn(
        "MedNeXt is not installed. Install it with "
        "`pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
    )

# MedNeXt-B presets, copied from nnunet_mednext.create_mednextv1_base so that a
# model built here is identical to ``create_mednext_v1(..., model_id='B')``.
_BASE_EXP_R = [2, 3, 4, 4, 4, 4, 4, 3, 2]
_BASE_BLOCK_COUNTS = [2, 2, 2, 2, 2, 2, 2, 2, 2]


class SqueezeExcitation3d(nn.Module):
    """Channel attention: global-pool, bottleneck MLP, per-channel gate."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        w = x.mean(dim=(2, 3, 4))            # (B, C) global average pool
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class SpatialAttention3d(nn.Module):
    """Spatial attention: channel-wise avg+max descriptor, conv, spatial gate."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        w = self.gate(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class SkipAttention3d(nn.Module):
    """Channel attention followed by spatial attention, as used on each skip."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = SqueezeExcitation3d(channels, reduction=reduction)
        self.spatial = SpatialAttention3d(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class MedNeXtDoseAttention(_MedNeXt if _MedNeXt is not None else object):
    """MedNeXt-B with attention-gated skip connections and a ReLU dose head.

    :param in_channels: number of input channels (PortPy dose prediction uses 8)
    :param out_channels: number of output channels, 1 for a dose map
    :param n_channels: base filter count. 32 reproduces MedNeXt-B (10.5 M params)
    :param kernel_size: convolution kernel size; the UpKern schedule walks 3 -> 5 -> 7
    :param se_reduction: bottleneck ratio of the squeeze-and-excitation block
    :param spatial_kernel: kernel size of the spatial attention convolution
    :param exp_r: per-stage expansion ratios, defaults to the MedNeXt-B preset
    :param block_counts: per-stage block counts, defaults to the MedNeXt-B preset

    :Example:

    >>> net = MedNeXtDoseAttention(in_channels=8, out_channels=1)  # doctest: +SKIP
    >>> dose = net(ct_and_masks)                                   # doctest: +SKIP
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        n_channels: int = 32,
        kernel_size: int = 3,
        se_reduction: int = 16,
        spatial_kernel: int = 7,
        exp_r: Sequence[int] = None,
        block_counts: Sequence[int] = None,
    ):
        if _MedNeXt is None:
            raise ImportError(
                "Requested MedNeXtDoseAttention but MedNeXt is not installed. "
                "Install it with "
                "`pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
            )
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=out_channels,
            exp_r=list(exp_r) if exp_r is not None else list(_BASE_EXP_R),
            kernel_size=kernel_size,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=list(block_counts) if block_counts is not None
            else list(_BASE_BLOCK_COUNTS),
        )
        # one attention module per skip level, matching the encoder widths
        self.skip_attention = nn.ModuleList([
            SkipAttention3d(n_channels * (2 ** i), reduction=se_reduction,
                            spatial_kernel=spatial_kernel)
            for i in range(4)
        ])
        # dose is non-negative; MedNeXt's segmentation head has no activation
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, site: str = None) -> torch.Tensor:
        """Same dataflow as MedNeXt, with each skip gated before it is added.

        ``site`` is accepted and ignored: only multi-site models use it.
        """
        x = self.stem(x)

        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)

        # MedNeXt ADDS the skip, so gate the encoder feature before the add
        x = self.dec_block_3(self.skip_attention[3](x_res_3) + self.up_3(x))
        x = self.dec_block_2(self.skip_attention[2](x_res_2) + self.up_2(x))
        x = self.dec_block_1(self.skip_attention[1](x_res_1) + self.up_1(x))
        x = self.dec_block_0(self.skip_attention[0](x_res_0) + self.up_0(x))

        return self.out_act(self.out_0(x))