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

from __future__ import annotations
import os
import pickle
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from portpy.photon.plan import Plan


def save_plan(my_plan: Plan, plan_name: str = None, path: str = None) -> None:
    """

    Save pickled file for plan object

    :param my_plan: object fo class Plan
    :param plan_name: create the name of the pickled file of plan object. If none, it will save with the name as 'my_plan'
    :param path: if path is set, plan object will be pickled and saved in path directory else it will save in current project directory
    :return: save pickled object of class Plan

    :Example:
    >>> my_plan.save_plan(plan_name='my_plan', path=r"path/to/save_plan")
    """
    if path is None:
        path = os.getcwd()
    elif not os.path.exists(path):
        os.makedirs(path)

    if plan_name is None:
        plan_name = 'my_plan'
    with open(os.path.join(path, plan_name), 'wb') as pickle_file:
        # pickle the dictionary and write it to file
        pickle.dump(my_plan, pickle_file)

def save_obj_as_pickle(obj, obj_name: str = None, path: str = None):
    if path is None:
        path = os.getcwd()
    elif not os.path.exists(path):
        os.makedirs(path)

    if obj_name is None:
        obj_name = 'obj'
    with open(os.path.join(path, obj_name), 'wb') as pickle_file:
        # pickle the dictionary and write it to file
        pickle.dump(obj, pickle_file)

def load_pickle_as_obj(obj_name: str = None, path: str = None):
    if path is None:
        path = os.getcwd()
    elif not os.path.exists(path):
        os.makedirs(path)

    if obj_name is None:
        obj_name = 'obj'
    with open(os.path.join(path, obj_name), 'rb') as pickle_file:
        return pickle.load(pickle_file)


def load_plan(plan_name: str = None, path: str = None):
    """
    Load pickle file of the plan object.

    :param plan_name: plan_name of the object of class Plan. It None, it will try to look for plan name called 'my_plan'
    :param path: if path is set, plan object will be load from path directory else current project directory
    :return: load pickled object of class Plan

    :Example:
    >>> load_plan(plan_name='my_plan', path=r"path/for/loading_plan")
    """
    if path is None:
        path = os.getcwd()

    if plan_name is None:
        plan_name = 'my_plan'
    with open(os.path.join(path, plan_name), 'rb') as pickle_file:
        my_plan = pickle.load(pickle_file)
    return my_plan


def load_optimal_sol(sol_name: str, path: str = None) -> dict:
    """

    Load optimal solution dictionary got from optimization

    :param sol_name: name of the optimal solution to be loaded.
    :param path: if path is set, plan object will be load from path directory else current directory
    :return: load solution

    :Example:
    >>> sol = load_optimal_sol(sol_name='sol', path=r'path/for/loading_sol')
    """
    if path is None:
        path = os.getcwd()

    with open(os.path.join(path, sol_name), 'rb') as pickle_file:
        sol = pickle.load(pickle_file)
    return sol


def save_optimal_sol(sol: dict, sol_name: str, path: str = None) -> None:
    """
    Save the optimal solution dictionary from optimization

    :param sol: optimal solution dictionary
    :param sol_name: name of the optimal solution saved
    :param path: if path is set, plan object will be load from path directory else current directory
    :return: save pickled file of optimal solution dictionary

    :Example:
    >>> save_optimal_sol(sol=sol, sol_name='sol', path=r'path/to/save_solution')
    """
    if path is None:
        path = os.getcwd()
    elif not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, sol_name), 'wb') as pickle_file:
        pickle.dump(sol, pickle_file, protocol=4)


def save_inf_matrix(inf_matrix, inf_name: str = None, path: str = None) -> None:
    """

    Save pickled file for plan object

    :param inf_matrix: object fo class Infuence matrix
    :param inf_name: create the name of the pickled file of InfluenceMatrix object. If none, it will save with the name as 'inf_matrix'
    :param path: if path is set, plan object will be pickled and saved in path directory else it will save in current project directory
    :return: save pickled object of class Plan

    :Example:
    >>> save_inf_matrix(inf_matrix=inf_matrix, inf_name='inf_matrix', path=r"path/to/save_inf_matrix")
    """
    if path is None:
        path = os.getcwd()
    elif not os.path.exists(path):
        os.makedirs(path)

    if inf_name is None:
        inf_matrix = 'inf_matrix'
    with open(os.path.join(path, inf_name), 'wb') as pickle_file:
        # pickle the dictionary and write it to file
        pickle.dump(inf_matrix, pickle_file)


def load_inf_matrix(inf_name: str = None, path: str = None):
    """
    Load pickle file of the plan object.

    :param inf_name: influence matrix name of the object of class InfleunceMatrix.
    :param path: if path is set, plan object will be load from path directory else current project directory
    :return: load pickled object of class Plan

    :Example:
    >>> load_plan(plan_name='my_plan', path=r"path/for/loading_inf_matrix")
    """
    if path is None:
        path = os.getcwd()

    if inf_name is None:
        inf_name = 'inf_matrix'
    with open(os.path.join(path, inf_name), 'rb') as pickle_file:
        inf_matrix = pickle.load(pickle_file)
    return inf_matrix
