# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2019, Battelle Memorial Institute
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of the FreeBSD
# Project.
#
# This material was prepared as an account of work sponsored by an
# agency of the United States Government.  Neither the United States
# Government nor the United States Department of Energy, nor Battelle,
# nor any of their employees, nor any jurisdiction or organization that
# has cooperated in the development of these materials, makes any
# warranty, express or implied, or assumes any legal liability or
# responsibility for the accuracy, completeness, or usefulness or any
# information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights.
#
# Reference herein to any specific commercial product, process, or
# service by trade name, trademark, manufacturer, or otherwise does not
# necessarily constitute or imply its endorsement, recommendation, or
# favoring by the United States Government or any agency thereof, or
# Battelle Memorial Institute. The views and opinions of authors
# expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#
# PACIFIC NORTHWEST NATIONAL LABORATORY
# operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY
# under Contract DE-AC05-76RL01830
# }}}
"""Example optimizer builds MILP problem from component & forecast info"""
from collections import OrderedDict
import itertools
import logging

import pulp

from econ_dispatch.optimizer import get_pulp_optimization_function


LOG = logging.getLogger(__name__)


def get_optimization_function(config):
    """Use helper function from __init__ to handle PuLP interactions"""
    return get_pulp_optimization_function(build_problem, config)


def constant(value):
    """define constant function"""

    def _constant(*args, **kwargs):
        """return constant value"""
        return value

    return _constant


class Storage(object):
    """Wrap common data for all storage components"""

    def __init__(self, pmax, emax, eta_ch, eta_disch, soc_max=1.0, soc_min=0.0, now_soc=0.0, name=None):
        self.pmax = pmax
        self.emax = emax
        self.eta_ch = eta_ch
        self.eta_disch = eta_disch
        self.soc_max = soc_max
        self.soc_min = soc_min

        self.now_soc = now_soc
        if now_soc is None:
            raise ValueError("STATE OF CHARGE IS NONE")

        self.name = name


# variable to indicate that we want all variables that match a pattern
# one item in the tuple key can be RANGE
RANGE = -1


class VariableGroup(object):
    """Define a group of PuLP variables from multiple indices

    :param name: variable names are based on this name
    :param indexes: the product of these indices defines the group
    :param is_integer_var: whether to restrict domain to integers
    :param lower_bound_func: function which maps an index to a lower bound
        (default is 0)
    :param upper_bound_func: function which maps an index to a upper bound
    """

    def __init__(self, name, indexes=(), is_integer_var=False, lower_bound_func=constant(0), upper_bound_func=None):
        self.variables = {}

        name_base = name
        for _ in range(len(indexes)):
            name_base += "_{}"

        for index in itertools.product(*indexes):
            var_name = name_base.format(*index)
            var_type = pulp.LpInteger if is_integer_var else pulp.LpContinuous

            lower_bound = lower_bound_func(index) if lower_bound_func is not None else None
            upper_bound = upper_bound_func(index) if upper_bound_func is not None else None

            # the lower bound should be set if the upper bound is set
            if lower_bound is None and upper_bound is not None:
                raise RuntimeError("Lower bound should not be unset " "while upper bound is set")

            # create the MILP variable
            self.variables[index] = pulp.LpVariable(var_name, cat=var_type, lowBound=lower_bound, upBound=upper_bound)

    def match(self, key):
        """return variables matching full range of a single index"""
        position = key.index(RANGE)  # which index to skip

        def predicate(keys_0, keys_1):
            """whether all other indices match search key"""
            num_matching = 0
            for i, (k_0, k_1) in enumerate(zip(keys_0, keys_1)):
                if i != position and k_0 == k_1:
                    num_matching += 1
            return num_matching == len(key) - 1

        # all variables
        keys = list(self.variables.keys())
        # only those which match, including any from the RANGE index
        keys = [k for k in keys if predicate(k, key)]
        # sort along the RANGE index
        keys.sort(key=lambda k: k[position])

        return [self.variables[k] for k in keys]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        n_range = key.count(RANGE)

        if n_range == 0:
            return self.variables[key]
        elif n_range == 1:
            return self.match(key)
        else:
            raise ValueError("Can only get RANGE for one index.")


def build_problem(forecast, parameters):
    """Define optimization problem from forecasts and component parameters

    :param forecast: dynamic information that can change over time
    :type forecast: list[dict]
    :param parameters: static information that does not change over time
    :type parameters: dict
    :returns: optimization problem to be solved
    :rtype: PuLP mixed-integer linear program
    """
    # --------------------------------------------------------------------------
    # CONSTANTS
    # --------------------------------------------------------------------------

    # length of optimization window, in time steps
    NN = len(forecast)
    # big number for penalties which approximate constraints
    BIGM = 1e4

    # number of pieces in piecewise-linear efficiency curves
    KK = 5

    # --------------------------------------------------------------------------
    # PRE-PROCESS FORECASTS & PARAMETERS
    # --------------------------------------------------------------------------

    # convert forecast from list of dicts to dict of lists
    _forecast = {}
    for f_c in forecast:
        for key, value in list(f_c.items()):
            try:
                _forecast[key].append(value)
            except KeyError:
                _forecast[key] = [value]
    forecast = _forecast

    # # uncomment to log forecasts and parameters
    # LOG.debug("============================================================")
    # for k, v in forecast.items():
    #     LOG.debug((k, v))
    # LOG.debug("============================================================")
    # import json
    # LOG.debug(json.dumps(parameters, indent=4, sort_keys=True))
    # LOG.debug("============================================================")

    component_names = sorted(list(parameters["IOComponent"].keys()))
    component_para = OrderedDict()
    for name in component_names:
        component_para[name] = parameters["IOComponent"][name]

    storage_para = OrderedDict()
    elec_storage_names = sorted(list(parameters.get("battery", {}).keys()))
    for name in elec_storage_names:
        _params = parameters["battery"][name]
        storage_para[name] = Storage(
            emax=_params["capacity"],
            pmax=_params["max_power"],
            eta_ch=_params["charge_eff"],
            eta_disch=_params["discharge_eff"],
            soc_min=_params["min_soc"],
            soc_max=_params["max_soc"],
            now_soc=_params["soc"],
            name=name,
        )

    heat_storage_names = sorted(list(parameters.get("thermal_storage", {}).keys()))
    for name in heat_storage_names:
        _params = parameters["thermal_storage"][name]
        storage_para[name] = Storage(
            emax=_params["heat_cap"],
            pmax=_params["max_power"],
            eta_ch=_params["eff"],
            eta_disch=_params["eff"],
            now_soc=_params["soc"],
            name=name,
        )

    # --------------------------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------------------------

    index_hour = (list(range(NN)),)
    index_without_first_hour = (list(range(1, NN)),)

    index_storage = elec_storage_names + heat_storage_names, list(range(NN))
    index_component = component_names, list(range(NN))
    index_component_piecewise = component_names, list(range(NN)), list(range(KK))

    index_ramp_up = ([name for name in component_names if component_para[name]["ramp_up"] is not None],)
    index_ramp_down = ([name for name in component_names if component_para[name]["ramp_down"] is not None],)

    index_gas_in = [name for name in component_names if component_para[name]["input_commodity"] == "gas"]
    # no component outputs gas
    index_elec_in = [name for name in component_names if component_para[name]["input_commodity"] == "elec"]
    index_elec_out = [name for name in component_names if component_para[name]["output_commodity"] == "elec"]
    index_cool_in = [name for name in component_names if component_para[name]["input_commodity"] == "cool"]
    index_cool_out = [name for name in component_names if component_para[name]["output_commodity"] == "cool"]
    index_heat_in = [name for name in component_names if component_para[name]["input_commodity"] == "heat"]
    index_heat_out = [name for name in component_names if component_para[name]["output_commodity"] == "heat"]

    # --------------------------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------------------------

    # market commodities
    elec_from_grid = VariableGroup("elec_from_grid", indexes=index_hour)
    elec_to_grid = VariableGroup("elec_to_grid", indexes=index_hour)

    # include these in balances with high penalty to ensure feasibility
    # if these are nonzero, problem definition is likely wrong
    elec_unserve = VariableGroup("elec_unserve", indexes=index_hour)
    elec_dump = VariableGroup("elec_dump", indexes=index_hour)
    heat_unserve = VariableGroup("heat_unserve", indexes=index_hour)
    heat_dump = VariableGroup("heat_dump", indexes=index_hour)
    cool_unserve = VariableGroup("cool_unserve", indexes=index_hour)
    cool_dump = VariableGroup("cool_dump", indexes=index_hour)

    # for heat recovery unit accounting
    heat_hru_out = VariableGroup("heat_hru_out", indexes=index_hour)

    # IO Components
    component_input = VariableGroup("component_input", indexes=index_component)
    component_output = VariableGroup("component_output", indexes=index_component)
    component_output_k = VariableGroup("component_output_k", indexes=index_component_piecewise)
    component_status = VariableGroup(
        "component_status", indexes=index_component, upper_bound_func=constant(1), is_integer_var=True
    )
    component_status_k = VariableGroup(
        "component_status_k", indexes=index_component_piecewise, upper_bound_func=constant(1), is_integer_var=True
    )
    component_start = VariableGroup(
        "component_start", indexes=index_component, upper_bound_func=constant(1), is_integer_var=True
    )

    # storage
    def storage_upper_bound(index):
        """storage (dis)charge upper bound is max power"""
        i = index[0]
        return storage_para[i].pmax

    storage_disch = VariableGroup("storage_disch", indexes=index_storage, upper_bound_func=storage_upper_bound)
    storage_ch = VariableGroup("storage_ch", indexes=index_storage, upper_bound_func=storage_upper_bound)

    def storage_state_lb(index):
        """storage state lower bound is minimum state of charge * capacity"""
        i = index[0]
        return storage_para[i].emax * storage_para[i].soc_min

    def storage_state_ub(index):
        """storage state upper bound is maximum state of charge * capacity"""
        i = index[0]
        return storage_para[i].emax * storage_para[i].soc_max

    storage_state = VariableGroup(
        "storage_state", indexes=index_storage, lower_bound_func=storage_state_lb, upper_bound_func=storage_state_ub
    )

    # --------------------------------------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------------------------------------

    # constraints should take the form `(con, name)` where `con` is a boolean
    # statement containing PuLP variables and `name` is a string label

    constraints = []

    def add_constraint(name, indexes, constraint_func):
        """Add a constraint for each index in a group

        :param name: base for constraint names
        :param indexes: iterable of indexes to apply constraints to
        :param constraint_func: applied to each element in product of indexes
        """
        name_base = name
        for _ in range(len(indexes)):
            name_base += "_{}"

        for index in itertools.product(*indexes):
            name = name_base.format(*index)
            con = constraint_func(index)

            constraints.append((con, name))

    #################
    # Energy Balances
    #################

    def elec_balance(index):
        """Balance electricity supply/demand for each timestep"""
        t = index[0]
        return (
            pulp.lpSum([component_output[i, t] for i in index_elec_out])
            - pulp.lpSum([component_input[i, t] for i in index_elec_in])
            + elec_from_grid[t]
            - elec_to_grid[t]
            + pulp.lpSum([storage_disch[i, t] for i in elec_storage_names])
            - pulp.lpSum([storage_ch[i, t] for i in elec_storage_names])
            + elec_unserve[t]
            - elec_dump[t]
            == forecast["elec_load"][t]
        )

    add_constraint("elec_balance", index_hour, elec_balance)

    def heat_balance(index):
        """Balance heat supply/demand for each timestep"""
        t = index[0]
        return (
            heat_hru_out[t]
            + pulp.lpSum([component_output[i, t] for i in index_heat_out])
            - pulp.lpSum([component_input[i, t] for i in index_heat_in])
            + heat_unserve[t]
            - heat_dump[t]
            == forecast["heat_load"][t]
        )

    add_constraint("heat_balance", index_hour, heat_balance)

    def cool_balance(index):
        """Balance cool supply/demand for each timestep"""
        t = index[0]
        return (
            pulp.lpSum([component_output[i, t] for i in index_cool_out])
            - pulp.lpSum([component_input[i, t] for i in index_cool_in])
            + pulp.lpSum([storage_disch[i, t] for i in heat_storage_names])
            - pulp.lpSum([storage_ch[i, t] for i in heat_storage_names])
            + cool_unserve[t]
            - cool_dump[t]
            == forecast["cool_load"][t]
        )

    add_constraint("cool_balance", index_hour, cool_balance)

    ##################
    # Storage Behavior
    ##################

    storage_start_state = {i: storage_para[i].now_soc * storage_para[i].emax for i in index_storage[0]}

    def storage_init(index):
        """Storage balance at first timestep"""
        i, t = index[0], 0
        return (
            storage_state[i, t]
            == storage_start_state[i]
            + storage_para[i].eta_ch * storage_ch[i, t]
            - 1 / storage_para[i].eta_disch * storage_disch[i, t]
        )

    add_constraint("storage_init", (index_storage[0],), storage_init)

    def storage_state_constraint(index):
        """Storage balance at 0<t<N-1"""
        i, t = index
        return (
            storage_state[i, t]
            == storage_state[i, t - 1]
            + storage_para[i].eta_ch * storage_ch[i, t]
            - 1 / storage_para[i].eta_disch * storage_disch[i, t]
        )

    add_constraint("storage_state_constraint", (index_storage[0],) + index_without_first_hour, storage_state_constraint)

    def storage_final(index):
        """Storage balance at last timestep

        Forces storage[N-1] >= storage[0]
        """
        i, t = index[0], NN - 1
        return storage_state[i, t] >= storage_start_state[i]

    add_constraint("storage_final", (index_storage[0],), storage_final)

    #######################
    # IO Component Behavior
    #######################

    def component_input_constraint(index):
        """Input is piecewise-linear function of output"""
        i, t = index
        return component_input[i, t] == [
            a * v for a, v in zip(component_para[i]["fundata"]["a"], component_output_k[i, t, RANGE])
        ] + [b * v for b, v in zip(component_para[i]["fundata"]["b"], component_status_k[i, t, RANGE])]

    add_constraint("component_input_constraint", index_component, component_input_constraint)

    def component_output_constraint(index):
        """Output is sum of output pieces"""
        i, t = index
        return component_output[i, t] == pulp.lpSum(component_output_k[i, t, RANGE])

    add_constraint("component_output_constraint", index_component, component_output_constraint)

    def component_piece_lower(index):
        """Lower bounds of output pieces"""
        i, t, k = index
        xmin = component_para[i]["fundata"]["min"][k]
        return xmin * component_status_k[i, t, k] <= component_output_k[i, t, k]

    add_constraint("component_piece_lower", index_component_piecewise, component_piece_lower)

    def component_piece_upper(index):
        """Upper bounds of output pieces"""
        i, t, k = index
        xmax = component_para[i]["fundata"]["max"][k]
        return xmax * component_status_k[i, t, k] >= component_output_k[i, t, k]

    add_constraint("component_piece_upper", index_component_piecewise, component_piece_upper)

    def component_status_constraint(index):
        """Status is sum of status pieces"""
        i, t = index
        return component_status[i, t] == pulp.lpSum(component_status_k[i, t, RANGE])

    add_constraint("component_status_constraint", index_component, component_status_constraint)

    def component_start_status_init(index):
        """Whether component starts up on first timestep"""
        i, t = index[0], 0
        status = component_para[i]["command_history"][-1]
        return component_start[i, t] >= component_status[i, t] - status

    add_constraint("component_start_status_init", (component_names,), component_start_status_init)

    def component_start_status(index):
        """Whether component starts up on t>0"""
        i, t = index
        return component_start[i, t] >= component_status[i, t] - component_status[i, t - 1]

    add_constraint("component_start_status", (component_names,) + index_without_first_hour, component_start_status)

    def component_ramp_up_init(index):
        """Do not increase output by too much on first timestep"""
        i, t = index[0], 0
        ramp_up = component_para[i]["ramp_up"]
        output_init = component_para[i]["output"]
        return component_output[i, t] <= output_init + ramp_up

    add_constraint("component_ramp_up_init", index_ramp_up, component_ramp_up_init)

    def component_ramp_up(index):
        """Do not increase output by too much for t>0"""
        i, t = index
        ramp_up = component_para[i]["ramp_up"]
        return component_output[i, t] <= component_output[i, t - 1] + ramp_up

    add_constraint("component_ramp_up", index_ramp_up + index_without_first_hour, component_ramp_up)

    def component_ramp_down_init(index):
        """Ramp down constraint on first timestep"""
        i, t = index[0], 0
        ramp_down = component_para[i]["ramp_down"]
        output_init = component_para[i]["output"]
        return output_init + ramp_down <= component_output[i, t]

    add_constraint("component_ramp_down_init", index_ramp_down, component_ramp_down_init)

    def component_ramp_down(index):
        """Do not decrease output by too much for t>0"""
        i, t = index
        ramp_down = component_para[i]["ramp_down"]
        return component_output[i, t - 1] + ramp_down <= component_output[i, t]

    add_constraint("component_ramp_down", index_ramp_down + index_without_first_hour, component_ramp_down)

    # it was easier to define these constraints in a for-loop instead of with
    # the `add_constraint` function, but either could be used.
    name = "component_lock_on_{}_{}"
    for i in component_names:
        history = component_para[i]["command_history"]
        min_on = component_para[i]["min_on"]
        if min_on == 0:
            continue
        # history all from parameters
        for t in range(0, 1):
            con = min_on * (history[-1] - component_status[i, t]) <= pulp.lpSum(history[NN + t - min_on : NN])
            constraints.append((con, name.format(i, t)))
        # history partially from parameters, partially decision
        for t in range(1, min_on):
            con = min_on * (component_status[i, t - 1] - component_status[i, t]) <= pulp.lpSum(
                [component_status[i, tau] for tau in range(0, t)]
            ) + pulp.lpSum(history[NN + t - min_on : NN])
            constraints.append((con, name.format(i, t)))
        # history all decision
        for t in range(min_on, NN):
            con = min_on * (component_status[i, t - 1] - component_status[i, t]) <= pulp.lpSum(
                [component_status[i, tau] for tau in range(t - min_on, t)]
            )
            constraints.append((con, name.format(i, t)))

    name = "component_lock_off_{}_{}"
    for i in component_names:
        history = component_para[i]["command_history"]
        min_off = component_para[i]["min_off"]
        if min_off == 0:
            continue
        # history all from parameters
        for t in range(0, 1):
            con = min_off * (component_status[i, t] - history[-1]) <= pulp.lpSum(
                [1 - v for v in history[NN + t - min_off : NN]]
            )
            constraints.append((con, name.format(i, t)))
        # history partially from parameters, partially decision
        for t in range(1, min_off):
            con = min_off * (component_status[i, t] - component_status[i, t - 1]) <= pulp.lpSum(
                [1 - component_status[i, tau] for tau in range(0, t)]
            ) + pulp.lpSum([1 - v for v in history[NN + t - min_off : NN]])
            constraints.append((con, name.format(i, t)))
        # history all decision
        for t in range(min_off, NN):
            con = min_off * (component_status[i, t] - component_status[i, t - 1]) <= pulp.lpSum(
                [1 - component_status[i, tau] for tau in range(t - min_off, t)]
            )
            constraints.append((con, name.format(i, t)))

    def hru_limit(index):
        """Heat recovery unit behavior"""
        t = index[0]
        partial = []
        for name in component_names:
            eff = component_para[name]["hru_eff"]
            if eff == 0:
                continue
            convert = component_para[name]["hru_convert"]
            partial.append(eff * (component_input[name, t] - convert * component_output[name, t]))
        return heat_hru_out[t] <= pulp.lpSum(partial)

    add_constraint("hru_limit", index_hour, hru_limit)

    # --------------------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # --------------------------------------------------------------------------

    objective_components = []

    for var, _lambda in zip(elec_from_grid[RANGE], forecast["electricity_cost"]):
        objective_components.append(var * _lambda)

    # # uncomment to sell electricity at purchase price
    # for var, _lambda in zip(elec_to_grid[RANGE],
    #                         forecast["electricity_cost"]):
    #     objective_components.append(var * _lambda * -1.)

    for i in index_gas_in:
        for var, _lambda in zip(component_input[i, RANGE], forecast["natural_gas_cost"]):
            objective_components.append(var * _lambda)

    for i in component_names:
        for var in component_start[i, RANGE]:
            objective_components.append(var * component_para[i]["start_cost"])

    for i in component_names:
        for var in component_status[i, RANGE]:
            objective_components.append(var * component_para[i]["run_cost"])

    for group in (elec_unserve, elec_dump, heat_unserve, heat_dump, cool_unserve, cool_dump):
        for var in group[RANGE]:
            objective_components.append(var * BIGM)

    # --------------------------------------------------------------------------
    # BUILD PROBLEM
    # --------------------------------------------------------------------------

    prob = pulp.LpProblem("Economic Dispatch Optimization", pulp.LpMinimize)
    prob += pulp.lpSum(objective_components), "Objective Function"

    for con in constraints:
        try:
            prob += con
        except TypeError as ex:
            LOG.error(con)
            LOG.error(type(con[0]))
            LOG.error("PuLP variable problem with constraint {}: {}" "".format(con[1], con[0]))
            raise ex

    return prob
