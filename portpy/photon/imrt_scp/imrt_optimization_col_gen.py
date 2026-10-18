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

from portpy.photon.vmat_scp.vmat_optimization_col_gen import VmatOptimizationColGen


class ImrtOptimizationColGen(VmatOptimizationColGen):
    """Column generation that seeds the apertures of a multi-aperture IMRT plan.

    Run this before :class:`~portpy.photon.imrt_scp.imrt_scp_optimization.ImrtScpOptimization`
    when ``opt_params['opt_parameters']['initial_leaf_pos']`` is ``'CG'``, on the SAME
    plan -- the one returned by
    :func:`~portpy.photon.imrt_scp.imrt_apertures.create_imrt_plan`. The leaf positions
    it finds are left on that plan's arcs, which is what the SCP then refines.

    Seeding matters more for IMRT than it does for VMAT. ``BEV`` opens every aperture to
    the same beam's-eye view of the target, so all apertures of a beam would start
    IDENTICAL and the extra apertures buy nothing. Column generation prices one aperture
    at a time against the dose still missing, so with each replica presented as its own
    beam every aperture gets a different shape.

    Nothing is overridden. The algorithm already works per beam, and the MU scaling that
    ``ImrtScpOptimization`` has to neutralise does not exist on this path. The class is
    here so the IMRT workflow has its own name, and so anything IMRT-specific has one
    obvious home later.

    :param my_plan: object of class Plan, as returned by ``create_imrt_plan``
    :param opt_params: optimization parameters (as loaded by
        ``DataExplorer.load_config_opt_params``)

    :Example:

    >>> imrt_plan = create_imrt_plan(my_plan, num_apertures=3)   # doctest: +SKIP
    >>> cg_opt = ImrtOptimizationColGen(my_plan=imrt_plan, opt_params=opt_params)   # doctest: +SKIP
    >>> sol = cg_opt.run_col_gen_algo(solver='MOSEK', accept_unknown=True)   # doctest: +SKIP
    """
    pass