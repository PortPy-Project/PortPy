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

import copy

import numpy as np
import scipy.sparse as sp

from portpy.photon.plan import Plan
from portpy.photon.vmat_scp.arcs import Arcs


def create_imrt_plan(my_plan: Plan, num_apertures: int = 3) -> Plan:
    """Turn a plan into a multi-aperture IMRT plan the VMAT optimizers can solve.

    VMAT delivers ONE aperture per control point. IMRT step-and-shoot delivers
    ``num_apertures`` apertures per beam, and a beamlet's fluence is the SUM of the
    apertures that leave it open.

    ``vmat_scp`` cannot express that directly: :class:`~portpy.photon.vmat_scp.arcs.Arcs`
    assumes each aperture owns its own contiguous, non-overlapping block of influence
    matrix columns, and it writes beamlet weights by ASSIGNMENT rather than by
    accumulation. Listing the same beam several times in an arc therefore makes the
    apertures overwrite one another instead of summing.

    The way round it is to give every aperture a column block of its own: each beam's
    columns are REPLICATED ``num_apertures`` times, and each replica is presented as a
    separate beam. The per-beam invariant is restored, the apertures land in disjoint
    slots, and ``A @ w`` sums them for free, with no change to ``arcs.py``.

    The returned plan carries its own Beams, InfluenceMatrix and Arcs, built on PRIVATE
    copies. The plan you pass in is never modified, so it stays usable for evaluation
    and visualization. Build the optimization structures (RINDs) BEFORE calling this, so
    the replicated influence matrix already knows about them.

    Both IMRT optimizers take the plan this returns, exactly as the VMAT workflow feeds
    one plan to :class:`~portpy.photon.vmat_scp.vmat_optimization_col_gen.VmatOptimizationColGen`
    and :class:`~portpy.photon.vmat_scp.vmat_scp_optimization.VmatScpOptimization`.

    :param my_plan: object of class Plan
    :param num_apertures: apertures delivered per beam; 1 gives one aperture per beam,
        which is a useful smoke test but not a realistic IMRT plan
    :return: object of class Plan whose beams are the aperture replicas

    :Example:

    >>> imrt_plan = create_imrt_plan(my_plan, num_apertures=3)   # doctest: +SKIP
    >>> cg_opt = ImrtOptimizationColGen(my_plan=imrt_plan, opt_params=opt_params)   # doctest: +SKIP
    >>> sol = cg_opt.run_col_gen_algo(solver='MOSEK')   # doctest: +SKIP
    >>> scp_opt = ImrtScpOptimization(my_plan=imrt_plan, opt_params=opt_params)   # doctest: +SKIP
    >>> convergence = scp_opt.run_sequential_cvx_algo(solver='MOSEK')   # doctest: +SKIP
    """
    num_apertures = int(num_apertures)
    if num_apertures < 1:
        raise ValueError('num_apertures must be >= 1, got {}'.format(num_apertures))

    beams = _replicate_beams(my_plan.beams, num_apertures)
    inf_matrix = _replicate_inf_matrix(my_plan.inf_matrix, beams, num_apertures)
    arcs = Arcs(arcs_dict=_build_arcs_dict(beams), inf_matrix=inf_matrix)
    return Plan(ct=my_plan.ct, structs=my_plan.structures, beams=beams,
                inf_matrix=inf_matrix, clinical_criteria=my_plan.clinical_criteria,
                arcs=arcs)


def _replicate_beams(beams, num_apertures: int):
    """Private copy of ``beams`` with every beam repeated ``num_apertures`` times.

    Every entry of ``beams_dict`` is a per-beam list of the same length, so each is
    expanded by repeating its elements. Only the ids are regenerated, because
    ``Arcs`` looks beams up by id and duplicates would alias.

    :param beams: object of class Beams
    :param num_apertures: apertures delivered per beam
    :return: a deep copy whose beams_dict holds ``num_beams * num_apertures`` entries
    """
    beams_rep = copy.deepcopy(beams)
    bd = beams_rep.beams_dict
    n_beams = len(bd['ID'])
    k = num_apertures

    for key, value in bd.items():
        if isinstance(value, list) and len(value) == n_beams:
            bd[key] = [copy.deepcopy(value[i]) for i in range(n_beams) for _ in range(k)]

    # fresh ids on the private copy, so they cannot collide with the caller's
    bd['ID'] = list(range(n_beams * k))
    # which physical beam each aperture came from. Column generation reads this to avoid
    # picking several apertures of the SAME beam in one round: they share geometry, so they
    # would tie on score and come back with identical leaf positions.
    bd['aperture_group'] = [i for i in range(n_beams) for _ in range(k)]
    return beams_rep


def _replicate_inf_matrix(inf_matrix, beams_rep, num_apertures: int):
    """Private influence matrix whose columns are replicated per aperture.

    Column order is beam-major: all apertures of beam 0, then of beam 1, and so on,
    which keeps each arc a single ascending range as ``Arcs.preprocess`` expects.

    :param inf_matrix: object of class InfluenceMatrix
    :param beams_rep: the replicated Beams from :func:`_replicate_beams`
    :param num_apertures: apertures delivered per beam
    :return: a shallow copy carrying the widened A and the per-aperture beamlets_dict
    """
    k = num_apertures
    # shallow copy: the voxel-side data is read-only here and can be large
    inf_rep = copy.copy(inf_matrix)
    inf_rep._beams = beams_rep

    cols, beamlets_rep = [], []
    next_col = 0
    for beam_idx, beamlets in enumerate(inf_matrix.beamlets_dict):
        grid = np.asarray(beamlets['beamlet_idx_2d_finest_grid'])
        start = int(grid[grid >= 0].min())
        end = int(grid.max())
        block = np.arange(start, end + 1)

        for ap in range(k):
            entry = copy.deepcopy(beamlets)
            entry['beam_id'] = beams_rep.beams_dict['ID'][beam_idx * k + ap]

            # renumber the BEV grid into this aperture's own column block
            new_grid = grid.copy()
            inside = new_grid >= 0
            new_grid[inside] = new_grid[inside] - start + next_col
            entry['beamlet_idx_2d_finest_grid'] = new_grid
            entry['start_beamlet_idx'] = next_col
            entry['end_beamlet_idx'] = next_col + block.size - 1

            beamlets_rep.append(entry)
            cols.append(block)
            next_col += block.size

    col_idx = np.concatenate(cols)
    A = inf_matrix.A
    inf_rep.A = A[:, col_idx] if sp.issparse(A) else np.asarray(A)[:, col_idx]
    inf_rep.beamlets_dict = beamlets_rep
    return inf_rep


def _build_arcs_dict(beams_rep):
    """One arc holding every aperture, in column order.

    IMRT has no arcs; this only satisfies the container ``Arcs`` expects. A single
    arc keeps ``arc['start_beamlet_idx'] .. arc['end_beamlet_idx']`` one contiguous
    ascending range, which ``Arcs.preprocess`` relies on.

    :param beams_rep: the replicated Beams from :func:`_replicate_beams`
    :return: arcs_dict in the form Arcs accepts
    """
    return {'arcs': [{'arc_id': '01',
                      'beam_ids': list(beams_rep.beams_dict['ID'])}]}