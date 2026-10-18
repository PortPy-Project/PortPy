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
"""Deliverability-constrained IMRT optimization (step-and-shoot, multi-aperture).

``vmat_scp`` optimises ONE aperture per control point. IMRT step-and-shoot delivers
several apertures per beam and the fluence of a beamlet is the SUM of the apertures
that leave it open. This package reuses the VMAT sequential-convex machinery for that
case.

Prepare the plan once, then hand it to both optimizers, exactly as the VMAT workflow
does::

    imrt_plan = create_imrt_plan(my_plan, num_apertures=3)
    sol = ImrtOptimizationColGen(my_plan=imrt_plan, opt_params=p).run_col_gen_algo(...)
    convergence = ImrtScpOptimization(my_plan=imrt_plan, opt_params=p).run_sequential_cvx_algo(...)
"""
from portpy.photon.imrt_scp.imrt_apertures import create_imrt_plan
from portpy.photon.imrt_scp.imrt_optimization_col_gen import ImrtOptimizationColGen
from portpy.photon.imrt_scp.imrt_scp_optimization import ImrtScpOptimization

__all__ = ['create_imrt_plan', 'ImrtOptimizationColGen', 'ImrtScpOptimization']
