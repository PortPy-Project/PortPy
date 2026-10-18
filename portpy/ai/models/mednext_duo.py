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
"""Dual-decoder MedNeXt for 3D dose prediction (GDP-HMM Top-3 solution).

Reimplementation of the 3rd-place solution of the AAPM 2025 GDP-HMM Grand
Challenge:

    Ruochen Gao, Marius Staring and Frank Dankers, "MedNeXt-Duo: A Lightweight
    Dual-Decoder MedNeXt for Generalizable 3D Dose Prediction", Leiden
    University Medical Center, 2025.

The idea is one **shared encoder** feeding two **site-specific decoders**, so a
single model covers two treatment sites: the encoder learns what generalises and
each decoder specialises. The original targets head-and-neck and lung; PortPy's
two sites are lung and prostate, so the decoders are keyed by name rather than
hard-coded. Each decoder also gets its own output head.

The model is deliberately lightweight: the authors reduce the base filter count
from MedNeXt-B's 32, which offsets the cost of the second decoder.

    n_channels=32 -> 10.53 M params (MedNeXt-B)
    n_channels=16 ->  2.67 M        (the authors' released code)
    n_channels=12 ->  1.52 M        (the figure quoted in their report)

Their report says 12 and their code forces 16; we default to 16 and leave it as
a constructor argument.

Training alternates between decoders: each step updates the shared encoder plus
one decoder, with the sampling split across sites controlled by the caller.

Implementation note: the FIRST site reuses the decoder that
:class:`nnunet_mednext.MedNeXt` already builds, so its parameter names are
unchanged and a single-site MedNeXt checkpoint can warm-start that branch. Extra
sites get their own copies under ``extra_decoders``.
"""
from __future__ import annotations

import copy
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

# MedNeXt-B presets, copied from nnunet_mednext.create_mednextv1_base.
_BASE_EXP_R = [2, 3, 4, 4, 4, 4, 4, 3, 2]
_BASE_BLOCK_COUNTS = [2, 2, 2, 2, 2, 2, 2, 2, 2]

_DECODER_PARTS = ('up_3', 'dec_block_3', 'up_2', 'dec_block_2',
                  'up_1', 'dec_block_1', 'up_0', 'dec_block_0', 'out_0')


class MedNeXtDuo(_MedNeXt if _MedNeXt is not None else object):
    """MedNeXt with one shared encoder and one decoder per treatment site.

    :param in_channels: number of input channels (PortPy dose prediction uses 8)
    :param out_channels: number of output channels, 1 for a dose map
    :param n_channels: base filter count; 16 is the authors' lightweight setting
    :param kernel_size: convolution kernel size
    :param sites: decoder names, one per site. The first reuses MedNeXt's own
        decoder so its parameter names stay compatible with a stock checkpoint
    :param exp_r: per-stage expansion ratios, defaults to the MedNeXt-B preset
    :param block_counts: per-stage block counts, defaults to the MedNeXt-B preset

    :Example:

    >>> net = MedNeXtDuo(in_channels=8, sites=('lung', 'prostate'))  # doctest: +SKIP
    >>> dose = net(x, site='prostate')                               # doctest: +SKIP
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        n_channels: int = 16,
        kernel_size: int = 3,
        sites: Sequence[str] = ('lung', 'prostate'),
        exp_r: Sequence[int] = None,
        block_counts: Sequence[int] = None,
    ):
        if _MedNeXt is None:
            raise ImportError(
                "Requested MedNeXtDuo but MedNeXt is not installed. Install it "
                "with `pip install git+https://github.com/MIC-DKFZ/MedNeXt.git`."
            )
        sites = tuple(sites)
        if len(sites) < 2 or len(set(sites)) != len(sites):
            raise ValueError(
                "sites must be at least two distinct names, got {}".format(sites))
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
        self.sites = sites
        # sites[0] uses the inherited decoder; the rest get their own copies.
        # init_net re-initialises every weight afterwards, so copying here only
        # clones the module STRUCTURE, never the values.
        self.extra_decoders = nn.ModuleDict({
            name: nn.ModuleDict({p: copy.deepcopy(getattr(self, p))
                                 for p in _DECODER_PARTS})
            for name in sites[1:]
        })

    def _decoder(self, site: str):
        """Return the decoder module set for ``site`` as a name -> module dict."""
        if site == self.sites[0]:
            return {p: getattr(self, p) for p in _DECODER_PARTS}
        try:
            return self.extra_decoders[site]
        except KeyError:
            raise KeyError("unknown site {!r}; expected one of {}".format(
                site, self.sites))

    def forward(self, x: torch.Tensor, site: str = None) -> torch.Tensor:
        """Encode once, then decode through the branch belonging to ``site``.

        :param x: input tensor
        :param site: which site decoder to use; defaults to the first site
        """
        dec = self._decoder(self.sites[0] if site is None else site)

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

        x = dec['dec_block_3'](x_res_3 + dec['up_3'](x))
        x = dec['dec_block_2'](x_res_2 + dec['up_2'](x))
        x = dec['dec_block_1'](x_res_1 + dec['up_1'](x))
        x = dec['dec_block_0'](x_res_0 + dec['up_0'](x))

        return dec['out_0'](x)