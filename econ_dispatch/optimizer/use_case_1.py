# -*- coding: utf-8 -*- {{{
# vim: set fenc = utf-8 ft = python sw = 4 ts = 4 sts = 4 et:

# Copyright (c) 2017, Battelle Memorial Institute
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

import numpy as np
import pandas as pd

import pulp
from pulp import LpVariable

from pprint import pformat

import logging
_log = logging.getLogger(__name__)

UNSERVE_LIMIT = 10000

def binary_var(name):
    return LpVariable(name, 0, 1, pulp.LpInteger)


class ChillerIGV(object):
    def __init__(self, name=None, xmin_chiller=None, a_chiller=None, b_chiller=None, xmax_chiller=None):
        self.name = name
        self.xmin_chiller = xmin_chiller
        self.a_chiller = a_chiller
        self.b_chiller = b_chiller
        self.xmax_chiller = xmax_chiller

        self.state_variables = {}
        self.chiller_elec = {}
        self.q_chiller = {}
        self.q_chiller_aux = {}

    def get_state_variable(self, hour):
        try:
            Schiller = self.state_variables[hour]
        except KeyError:
            Schiller = binary_var("Schiller{}_hour{}".format(self.name, hour))
            self.state_variables[hour] = Schiller

        return Schiller

    def get_e_chiller_elec(self, hour):
        try:
            E_chillerelec = self.chiller_elec[hour]
        except KeyError:
            E_chillerelec = LpVariable("E_chillerelec{}_hour{}".format(self.name, hour), 0)
            print "E_chillerelec{}_hour{}".format(self.name, hour)
            self.chiller_elec[hour] = E_chillerelec

        return E_chillerelec

    def get_q_chiller(self, hour):
        try:
            Q_chiller = self.q_chiller[hour]
        except KeyError:
            Q_chiller = LpVariable("Q_chiller{}_hour{}".format(self.name, hour), 0)
            self.q_chiller[hour] = Q_chiller

        return Q_chiller

    def get_q_chiller_aux(self, hour):
        try:
            Q_chiller_aux = self.q_chiller_aux[hour]
        except KeyError:
            Q_chiller_aux = LpVariable("Q_chiller{}_hour{}_aux{}".format(self.name, hour, 1), 0, self.xmax_chiller - self.xmin_chiller)
            self.q_chiller_aux[hour] = Q_chiller_aux

        return Q_chiller_aux

    def get_constraints(self, hour):
        Schiller = self.get_state_variable(hour)
        E_chillerelec = self.get_e_chiller_elec(hour)
        Q_chiller = self.get_q_chiller(hour)
        Q_chiller_aux = self.get_q_chiller_aux(hour)

        constraints = []

        label = "ChillerElecConsume{}_{}".format(self.name, hour)
        exp = E_chillerelec - self.a_chiller * Q_chiller_aux
        exp = exp - self.b_chiller * Schiller
        exp = exp == 0
        constraints.append((exp, label))

        label = "ChillerCoolGenerate{}_{}".format(self.name, hour)
        exp = Q_chiller - Q_chiller_aux
        exp = exp - self.xmin_chiller * Schiller
        exp = exp == 0
        constraints.append((exp, label))

        label = "ChillerQlower{}_{}".format(self.name, hour)
        exp = Q_chiller - self.xmin_chiller * Schiller >= 0
        constraints.append((exp, label))

        label = "ChillerQupper{}_{}".format(self.name, hour)
        exp = Q_chiller - self.xmax_chiller * Schiller <= 0
        constraints.append((exp, label))

        return constraints


def get_optimization_problem(forecast, parameters = {}):
    # get the model parameters and bounds for variables
    # load FuelCellPara.mat
    _log.debug("Parameters:\n"+pformat(parameters))

    try:
        fuel_cell_params = parameters["fuel_cell"]
        boiler_params = parameters["boiler"]
        centrifugal_chiller_igv_params = parameters["centrifugal_chiller_igv"]
        absorption_chiller_params = parameters["absorption_chiller"]
    except KeyError:
        raise RuntimeError("Missing needed configuration parameter: " + e.message)

    # prime mover (fuel cell/micro turbine generator)
    for fuel_cell_name, parameters in fuel_cell_params.items():
        mat_prime_mover = parameters["mat_prime_mover"]
        cap_prime_mover = parameters["cap_prime_mover"]

        xmin_prime_mover = cap_prime_mover * 0.3
        a_E_primer_mover = mat_prime_mover[1]
        b_E_prime_mover = mat_prime_mover[0] + a_E_primer_mover * xmin_prime_mover
        xmax_prime_mover = cap_prime_mover

        a_Q_primer_mover = a_E_primer_mover - 1 / 293.1
        b_Q_primer_mover = b_E_prime_mover - xmin_prime_mover / 293.1

    for boiler_name, parameters in boiler_params.items():
        mat_boiler = parameters["mat_boiler"]
        xmax_boiler =  parameters["xmax_boiler"]
        xmin_boiler =  parameters["xmin_boiler"]
        cap_boiler = parameters["cap_boiler"]
        xmin_boiler[0] = cap_boiler * 0.15 # !!!need to consider cases when xmin is not in the first section of the training data
        Nsection = np.where(xmax_boiler > cap_boiler)[0][0]
        Nsection = Nsection + 1
        xmin_boiler = xmin_boiler[:Nsection]
        xmax_boiler = xmax_boiler[:Nsection]
        a_boiler = mat_boiler[1][:Nsection]
        b_boiler = mat_boiler[0][0] + a_boiler[0] * xmin_boiler[0]
        xmax_boiler[-1] = cap_boiler

    chillers_igv = []
    for i, (name, parameters) in enumerate(centrifugal_chiller_igv_params.items()):
        mat_chillerIGV = parameters["mat_chillerIGV"]
        xmax_chillerIGV = parameters["xmax_chillerIGV"]
        xmin_chillerIGV = parameters["xmin_chillerIGV"]
        cap_chiller = parameters["capacity_per_chiller"]
        cap_chiller = cap_chiller * 3.517 / 293.1  # ton -> mmBtu/hr
        n_chiller = parameters["chiller_count"]

        xmin_chiller = cap_chiller * 0.15
        a_chiller = mat_chillerIGV[1] + i * 0.01
        b_chiller = mat_chillerIGV[0] + a_chiller * xmin_chillerIGV
        xmax_chiller = cap_chiller
        chiller = ChillerIGV(name=name,
                             xmin_chiller=xmin_chiller,
                             a_chiller=a_chiller,
                             b_chiller=b_chiller,
                             xmax_chiller=xmax_chiller)

        chillers_igv.append(chiller)

    for absorption_chiller_name, parameters in absorption_chiller_params.items():
        mat_abschiller = parameters["mat_abschiller"]
        xmax_abschiller = parameters["xmax_abschiller"]
        xmin_abschiller = parameters["xmin_abschiller"]
        min_on_abs_chiller = parameters.get("min_on_abs_chiller", 3)
        min_off_abs_chiller = parameters.get("min_off_abs_chiller", 0)
        abs_chiller_history = parameters["abs_chiller_history"]
        cap_abs = parameters["cap_abs_chiller"]
        cap_abs = cap_abs / 293.1  # kW -> mmBtu/hr

        # flagabs = True
        xmin_AbsChiller = cap_abs * 0.15
        a_abs = mat_abschiller[1]
        b_abs = mat_abschiller[0] + a_abs * xmin_AbsChiller
        xmax_AbsChiller = cap_abs

    # heat recovery unit
    a_hru = 0.8

    ################################################################################

    objective_component = []
    constraints = []
    absorption_chiller_state = []
    for hour, forecast_hour in enumerate(forecast):
        hour = str(hour).zfill(2)

        # binary variables
        Sturbine = binary_var("Sturbine_hour{}".format(hour))

        Sboiler = binary_var("Sboiler_hour{}".format(hour))

        Sabs = binary_var("Sabs_hour{}".format(hour))
        absorption_chiller_state.append(Sabs)

        # free variables
        E_gridelec = LpVariable("E_gridelec_hour{}".format(hour))

        # regular variables
        E_unserve = LpVariable("E_unserve_hour{}".format(hour), 0)
        E_dump = LpVariable("E_dump_hour{}".format(hour), 0)
        Heat_unserve = LpVariable("Heat_unserve_hour{}".format(hour), 0)
        Heat_dump = LpVariable("Heat_dump_hour{}".format(hour), 0)
        Cool_unserve = LpVariable("Cool_unserve_hour{}".format(hour), 0)
        Cool_dump = LpVariable("Cool_dump_hour{}".format(hour), 0)

        E_prime_mover_fuel = LpVariable("E_prime_mover_fuel_hour{}".format(hour), xmin_boiler[0])
        Q_prime_mover = LpVariable("Q_prime_mover_hour{}".format(hour), 0)
        E_prime_mover_elec = LpVariable("E_prime_mover_elec_hour{}".format(hour), 0)
        E_prime_mover_elec_aux = LpVariable("E_prime_mover_elec_hour{}_aux{}".format(hour, 1), 0, xmax_prime_mover - xmin_prime_mover)

        E_boilergas = LpVariable("E_boilergas_hour{}".format(hour), 0)

        Q_boiler = LpVariable("Q_boiler_hour{}".format(hour), 0)
        Q_boiler_aux = []
        for i in range(len(xmax_boiler)):
            var = LpVariable("Q_boiler_hour{}_aux{}".format(hour, i), 0, xmax_boiler[i] - xmin_boiler[i])
            Q_boiler_aux.append(var)

        Q_abs = LpVariable("Q_abs_hour{}".format(hour), 0)
        Q_abs_aux = [LpVariable("Q_abs_hour{}_aux0".format(hour), 0)]

        Q_HRUheating = LpVariable("Q_HRUheating_hour{}".format(hour), 0)
        Q_Genheating = LpVariable("Q_Genheating_hour{}".format(hour), 0)
        Q_Gencooling = LpVariable("Q_Gencooling_hour{}".format(hour), 0)

        # constraints
        objective_component += [
            forecast_hour["natural_gas_cost"]* E_prime_mover_fuel,
            forecast_hour["natural_gas_cost"] * E_boilergas,
            forecast_hour["electricity_cost"] * E_gridelec,
            UNSERVE_LIMIT * E_unserve,
            UNSERVE_LIMIT * E_dump,
            UNSERVE_LIMIT * Heat_unserve,
            UNSERVE_LIMIT * Heat_dump,
            UNSERVE_LIMIT * Cool_unserve,
            UNSERVE_LIMIT * Cool_dump
        ]

        # electric energy balance
        label = "ElecBalance{}".format(hour)
        exp = E_prime_mover_elec + E_gridelec
        for e_chill in [c.get_e_chiller_elec(hour) for c in chillers_igv]:
            exp = exp - e_chill
        exp = exp + E_unserve - E_dump
        exp = exp == forecast_hour["elec_load"] - forecast_hour["solar_kW"]
        constraints.append((exp, label))

        # heating balance
        label = "HeatBalance{}".format(hour)
        exp = Q_boiler + Q_HRUheating + Heat_unserve - Heat_dump == forecast_hour["heat_load"]
        constraints.append((exp, label))

        # cooling balance
        label = "CoolBalance{}".format(hour)
        exp = Q_abs
        for q_chill in [c.get_q_chiller(hour) for c in chillers_igv]:
            exp = exp + q_chill

        exp = exp + Cool_unserve - Cool_dump
        exp = exp == forecast_hour["cool_load"]
        constraints.append((exp, label))

        # generator gas
        label = "PrimeMoverFuelConsume{}".format(hour)
        exp = E_prime_mover_fuel - a_E_primer_mover * E_prime_mover_elec_aux - b_E_prime_mover * Sturbine == 0
        constraints.append((exp, label))

        # generator heat
        label = "PrimeMoverHeatGenerate{}".format(hour)
        exp = Q_prime_mover - a_Q_primer_mover * E_prime_mover_elec_aux - b_Q_primer_mover * Sturbine == 0
        constraints.append((exp, label))

        # microturbine elec
        label = "PrimeMoverElecGenerate{}".format(hour)
        exp = E_prime_mover_elec - E_prime_mover_elec_aux - xmin_prime_mover * Sturbine == 0
        constraints.append((exp, label))

        label = "PrimeMoverElower{}".format(hour)
        exp = E_prime_mover_elec - xmin_prime_mover * Sturbine >= 0
        constraints.append((exp, label))

        label = "PrimeMoverEupper{}".format(hour)
        exp = E_prime_mover_elec - xmax_prime_mover * Sturbine <= 0
        constraints.append((exp, label))

        # boiler
        label = "BoilerGasConsume{}".format(hour)
        exp = E_boilergas
        for a, q_boil in zip(a_boiler, Q_boiler_aux):
            exp = exp - a * q_boil
        exp = exp - b_boiler * Sboiler
        exp = exp == 0
        constraints.append((exp, label))

        label = "BoilerHeatGenerate{}".format(hour)
        exp = Q_boiler
        for q in Q_boiler_aux:
            exp = exp - q
        exp = exp - xmin_boiler[0] * Sboiler
        exp = exp == 0
        constraints.append((exp, label))

        label = "BoilerQlower{}".format(hour)
        exp = Q_boiler - xmin_boiler[0] * Sboiler >= 0
        constraints.append((exp, label))

        label = "BoilerQupper{}".format(hour)
        exp = Q_boiler - xmax_boiler[-1] * Sboiler <= 0
        constraints.append((exp, label))

        # chillers
        for chiller in chillers_igv:
            constraints.extend(chiller.get_constraints(hour))


        # abschiller
        label = "AbsChillerHeatCoolConsume{}".format(hour)
        exp = Q_Gencooling
        for q in Q_abs_aux:
            exp = exp - a_abs * q
        exp = exp - b_abs * Sabs
        exp = exp == 0
        constraints.append((exp, label))

        label = "AbsChillerHeatGenerate{}".format(hour)
        exp = Q_abs
        for q in Q_abs_aux:
            exp = exp - q
        exp = exp - xmin_AbsChiller * Sabs
        exp = exp == 0
        constraints.append((exp, label))

        label = "AbschillerQlower{}".format(hour)
        exp = Q_abs - xmin_AbsChiller * Sabs >= 0
        constraints.append((exp, label))

        label = "AbschillerQupper{}".format(hour)
        exp = Q_abs - xmax_AbsChiller * Sabs <= 0
        constraints.append((exp, label))

        # HRU
        label = "HRUWasteheat{}".format(hour)
        exp = Q_Genheating
        exp = exp + Q_Gencooling
        exp = exp - Q_prime_mover
        exp = exp == 0
        constraints.append((exp, label))

        label = "HRUHeatlimit{}".format(hour)
        exp = Q_HRUheating - a_hru * Q_Genheating <= 0
        constraints.append((exp, label))

    # time lock constraints need to look at multiple state variables
    def lock_on_constraints(label_template, min_lock_time, state_variables, state_history):
        # don't do anything if the lock is not needed
        if min_lock_time < 1:
            return

        window_states = state_history[-min_lock_time:] + state_variables
        window_size = min_lock_time + 1

        for hour in range(len(forecast)):
            window = window_states[:window_size]
            window_states = window_states[1:]
            current_time = window[-1]
            last_time = window[-2]

            exp = pulp.lpSum(window[:-1]) >= min_lock_time * (last_time - current_time)

            label = label_template.format(hour)
            constraints.append((exp, label))

    def lock_off_constraints(label_template, min_lock_time, state_variables, state_history):
        # don't do anything if the lock is not needed
        if min_lock_time < 1:
            return

        window_states = state_history[-min_lock_time:] + state_variables
        window_size = min_lock_time + 1

        for hour in range(len(forecast)):
            window = window_states[:window_size]
            window = [1 - x for x in window]

            window_states = window_states[1:]
            current_time = window[-1]
            last_time = window[-2]

            exp = pulp.lpSum(window[:-1]) >= min_lock_time * (last_time - current_time)

            label = label_template.format(hour)
            constraints.append((exp, label))

    lock_on_constraints("AbsLockOn{}", min_on_abs_chiller,
                        absorption_chiller_state, abs_chiller_history)


    # Build the optimization problem
    prob = pulp.LpProblem("Building Optimization", pulp.LpMinimize)

    objective_function = objective_component[0]
    for component in objective_component[1:]:
        objective_function += component

    prob += objective_function, "Objective Function"

    for c in constraints:
       prob += c

    return prob
