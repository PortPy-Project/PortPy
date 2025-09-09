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

# get smoothness function value using leaf positions in arcs
def get_apt_reg_metric(my_plan, beam_ids=None):
    if beam_ids is None:
        beam_ids = [beam['beam_id'] for arc in my_plan.arcs.arcs_dict['arcs'] for beam in arc['vmat_opt']]
    arcs = my_plan.arcs
    apt_reg_obj = 0
    for arc in arcs.arcs_dict['arcs']:
        for beam in arc['vmat_opt']:
            if beam['beam_id'] in beam_ids:
                num_rows = beam['num_rows']
                for r, row in enumerate(beam['reduced_2d_grid']):
                    if r < num_rows - 1:
                        curr_left = beam['leaf_pos_left'][r] + 1
                        curr_right = beam['leaf_pos_right'][r]
                        next_left = beam['leaf_pos_left'][r + 1] + 1
                        next_right = beam['leaf_pos_right'][r + 1]
                        left_diff = abs(curr_left - next_left)
                        right_diff = abs(curr_right - next_right)
                        apt_reg_obj += left_diff ** 2 + right_diff ** 2
    # print('####apr_reg_obj: {} ####'.format(apt_reg_obj))
    return apt_reg_obj