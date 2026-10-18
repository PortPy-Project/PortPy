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

import numpy as np

from portpy.photon.vmat_scp.vmat_scp_optimization import VmatScpOptimization


class ImrtScpOptimization(VmatScpOptimization):
    """Sequential convex programming for multi-aperture IMRT.

    Takes the plan returned by
    :func:`~portpy.photon.imrt_scp.imrt_apertures.create_imrt_plan`, whose beams are the
    aperture replicas, and is otherwise the VMAT SCP -- the leaf-position refinement and
    the interior/boundary beamlet split carry over unchanged.

    The one thing that MUST change is the monitor-unit scaling.
    :meth:`~portpy.photon.vmat_scp.vmat_scp_optimization.VmatScpOptimization.create_cvx_params`
    scales each control point's MU by half the gantry span to its neighbours,
    ``adjust_mu = min(diff, 360 - diff) / 2``. Apertures of one IMRT beam share a gantry
    angle, so that span is 0 and every aperture would deliver ZERO dose -- silently, with
    no error. A static field has no such sweep, so the factor is simply 1 here.

    Overriding it in a subclass keeps ``vmat_scp_optimization.py``, which is clinically
    deployed, untouched.

    :param my_plan: object of class Plan, as returned by ``create_imrt_plan``
    :param opt_params: optimization parameters (as loaded by
        ``DataExplorer.load_config_opt_params``)

    :Example:

    >>> imrt_plan = create_imrt_plan(my_plan, num_apertures=3)   # doctest: +SKIP
    >>> imrt_opt = ImrtScpOptimization(my_plan=imrt_plan, opt_params=opt_params)   # doctest: +SKIP
    >>> convergence = imrt_opt.run_sequential_cvx_algo(solver='MOSEK')   # doctest: +SKIP
    """

    def create_cvx_params(self, *args, **kwargs):
        """Build the cvxpy matrices, then neutralise the angular MU scaling.

        :return: whatever the base implementation returns (an aperture influence
            matrix when called for the actual-solution correction, otherwise None)
        """
        out = super().create_cvx_params(*args, **kwargs)
        for key in ('map_adj_int', 'map_adj_bound'):
            if key in self.cvxpy_params:
                self.cvxpy_params[key] = np.ones_like(self.cvxpy_params[key])
        # arcs.calculate_dose reads this one separately
        for arc in self.arcs.arcs_dict['arcs']:
            arc['map_adj_int'] = np.ones(arc['num_beams'])
        return out