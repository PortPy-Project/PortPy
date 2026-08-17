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
# ----------------------------------------------------------------------
"""
portpy.ai.inference -- the beamlet-dose PREDICTION pipeline (DICOM -> A).

Modules:
  ray_geometry           pure-numpy beam geometry (rotation matrices, ray patch,
                         d-channels, BEV/source/beamlet coordinates)
  gpu_resample           torch grid_sample resampling (CT <-> ray <-> points)
  patient_grid_geometry  precomputed beam geometry on the patient CT grid
  inf_matrix_predictor   BeamletDosePredictor / RayUNetPredictor /
                         PatientGridPredictor + build_ray_unet_predictor factory

Model ARCHITECTURES live in portpy.ai.models; training scaffolding
(train.py, dataset/, options/, ...) stays at the portpy.ai top level.

NOTE: importing this package pulls no torch -- torch is imported lazily inside
gpu_resample / the predictors, so geometry-only users stay lightweight.
"""
# Geometry is numpy-only: safe to re-export eagerly.
from .ray_geometry import (
    get_rotation_matrix,
    source_lps_from_beam,
    beamlet_center_lps,
    build_ray_grid_with_beam_axes,
)

__all__ = [
    'get_rotation_matrix',
    'source_lps_from_beam',
    'beamlet_center_lps',
    'build_ray_grid_with_beam_axes',
]