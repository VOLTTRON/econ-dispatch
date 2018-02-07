# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

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

        self.state_variable = {}
        self.chiller_elec = {}
        self.q_chiller = {}
        self.q_chiller_aux = {}

    def get_state_variable(self, hour):
        try:
            Schiller = self.state_variable[hour]
        except KeyError:
            Schiller = binary_var("Schiller{}_hour{}".format(self.name, hour))
            self.state_variable[hour] = Schiller

        return Schiller

    def get_e_chiller_elec(self, hour):
        try:
            E_chillerelec = self.chiller_elec[hour]
        except KeyError:
            E_chillerelec = LpVariable("E_chillerelec{}_hour{}".format(self.name, hour), 0)
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


class AbsChiller(object):
    def __init__(self, name=None,
                 xmin_abschiller=None,
                 a_abs=None,
                 b_abs=None,
                 xmax_abschiller=None,
                 history=None,
                 min_on_time=None,
                 min_off_time=None):
        self.name = name
        self.xmin_abschiller = xmin_abschiller
        self.a_abs = a_abs
        self.b_abs = b_abs
        self.xmax_abschiller = xmax_abschiller
        self.history = history
        self.min_on_time = min_on_time
        self.min_off_time = min_off_time

        self.state_variable = {}
        self.q_abs = {}
        self.q_abs_aux = {}

    def get_state_variable(self, hour):
        try:
            Sabs = self.state_variable[hour]
        except KeyError:
            Sabs = binary_var("Sabs{}_hour{}".format(self.name, hour))
            self.state_variable[hour] = Sabs

        return Sabs

    def get_q_abs(self, hour):
        try:
            Q_abs = self.q_abs[hour]
        except KeyError:
            Q_abs = LpVariable("Q_abs{}_hour{}".format(self.name, hour), 0)
            self.q_abs[hour] = Q_abs

        return Q_abs

    def get_q_abs_aux(self, hour):
        try:
            Q_abs_aux = self.q_abs_aux[hour]
        except KeyError:
            Q_abs_aux = [LpVariable("Q_abs{}_hour{}_aux0".format(self.name, hour), 0)]
            self.q_abs_aux[hour] = Q_abs_aux

        return Q_abs_aux

    def get_constraints(self, hour, Q_Gencooling):
        Sabs = self.get_state_variable(hour)
        Q_abs = self.get_q_abs(hour)
        Q_abs_aux = self.get_q_abs_aux(hour)

        constraints = []

        label = "AbsChillerHeatCoolConsume{}_{}".format(self.name, hour)
        exp = Q_Gencooling
        for q in Q_abs_aux:
            exp = exp - self.a_abs * q
        exp = exp - self.b_abs * Sabs
        exp = exp == 0
        constraints.append((exp, label))

        label = "AbsChillerHeatGenerate{}_{}".format(self.name, hour)
        exp = Q_abs
        for q in Q_abs_aux:
            exp = exp - q
        exp = exp - self.xmin_abschiller * Sabs
        exp = exp == 0
        constraints.append((exp, label))

        label = "AbschillerQlower{}_{}".format(self.name, hour)
        exp = Q_abs - self.xmin_abschiller * Sabs >= 0
        constraints.append((exp, label))

        label = "AbschillerQupper{}_{}".format(self.name, hour)
        exp = Q_abs - self.xmax_abschiller * Sabs <= 0
        constraints.append((exp, label))

        return constraints


class Boiler(object):
    def __init__(self, name=None, xmin_boiler=None, a_boiler=None, b_boiler=None, xmax_boiler=None):
        self.name = name
        self.xmin_boiler = xmin_boiler
        self.a_boiler = a_boiler
        self.b_boiler = b_boiler
        self.xmax_boiler = xmax_boiler
        self.state_variable = {}
        self.e_boilergas = {}
        self.q_boiler = {}
        self.q_boiler_aux = {}

    def get_state_variable(self, hour):
        try:
            Sboiler = self.state_variable[hour]
        except KeyError:
            Sboiler = binary_var("Sboiler{}_hour{}".format(self.name, hour))
            self.state_variable[hour] = Sboiler

        return Sboiler

    def get_e_boilergas(self, hour):
        try:
            E_boilergas = self.e_boilergas[hour]
        except KeyError:
            E_boilergas = LpVariable("E_boilergas{}_hour{}".format(self.name, hour), 0)
            self.e_boilergas[hour] = E_boilergas

        return E_boilergas

    def get_q_boiler(self, hour):
        try:
            Q_boiler = self.q_boiler[hour]
        except KeyError:
            Q_boiler = LpVariable("Q_boiler{}_hour{}".format(self.name, hour), 0)
            self.q_boiler[hour] = Q_boiler

        return Q_boiler

    def get_q_boiler_aux(self, hour):
        try:
            Q_boiler_aux = self.q_boiler_aux[hour]
        except KeyError:
            Q_boiler_aux = []
            for i in range(len(self.xmax_boiler)):
                var = LpVariable("Q_boiler{}_hour{}_aux{}".format(self.name, hour, i), 0, self.xmax_boiler[i] - self.xmin_boiler[i])
                Q_boiler_aux.append(var)

            self.q_boiler_aux[hour] = Q_boiler_aux

        return Q_boiler_aux

    def get_constraints(self, hour):
        Sboiler = self.get_state_variable(hour)
        E_boilergas = self.get_e_boilergas(hour)
        Q_boiler = self.get_q_boiler(hour)
        Q_boiler_aux = self.get_q_boiler_aux(hour)

        constraints = []

        label = "BoilerGasConsume{}_{}".format(self.name, hour)
        exp = E_boilergas
        for a, q_boil in zip(self.a_boiler, Q_boiler_aux):
            exp = exp - a * q_boil
        exp = exp - self.b_boiler * Sboiler
        exp = exp == 0
        constraints.append((exp, label))

        label = "BoilerHeatGenerate{}_{}".format(self.name, hour)
        exp = Q_boiler
        for q in Q_boiler_aux:
            exp = exp - q
        exp = exp - self.xmin_boiler[0] * Sboiler
        exp = exp == 0
        constraints.append((exp, label))

        label = "BoilerQlower{}_{}".format(self.name, hour)
        exp = Q_boiler - self.xmin_boiler[0] * Sboiler >= 0
        constraints.append((exp, label))

        label = "BoilerQupper{}_{}".format(self.name, hour)
        exp = Q_boiler - self.xmax_boiler[-1] * Sboiler <= 0
        constraints.append((exp, label))

        return constraints


class PrimeMover(object):
    def __init__(self,
                 name=None,
                 xmin_prime_mover=None,
                 a_E_primer_mover=None,
                 b_E_prime_mover=None,
                 xmax_prime_mover=None,
                 a_Q_primer_mover=None,
                 b_Q_primer_mover=None):

        self.name = name
        self.xmin_prime_mover = xmin_prime_mover
        self.a_E_primer_mover = a_E_primer_mover
        self.b_E_prime_mover = b_E_prime_mover
        self.xmax_prime_mover = xmax_prime_mover
        self.a_Q_primer_mover = a_Q_primer_mover
        self.b_Q_primer_mover = b_Q_primer_mover

        self.state_variable = {}
        self.e_prime_mover_fuel = {}
        self.q_prime_mover = {}
        self.e_prime_mover_elec = {}
        self.e_prime_mover_elec_aux = {}

    def get_state_variable(self, hour):
        try:
            Sturbine = self.state_variable[hour]
        except KeyError:
            Sturbine = binary_var("Sturbine{}_hour{}".format(self.name, hour))
            self.state_variable[hour] = Sturbine

        return Sturbine

    def get_e_prime_mover_fuel(self, hour, xmin_boiler):
        try:
            E_prime_mover_fuel = self.e_prime_mover_fuel[hour]
        except KeyError:
            E_prime_mover_fuel = LpVariable("E_prime_mover_fuel{}_hour{}".format(self.name, hour), xmin_boiler[0])
            self.e_prime_mover_fuel[hour] = E_prime_mover_fuel

        return E_prime_mover_fuel

    def get_q_prime_mover(self, hour):
        try:
            Q_prime_mover = self.q_prime_mover[hour]
        except KeyError:
            Q_prime_mover = LpVariable("Q_prime_mover{}_hour{}".format(self.name, hour), 0)
            self.q_prime_mover[hour] = Q_prime_mover

        return Q_prime_mover

    def get_e_prime_mover_elec(self, hour):
        try:
            E_prime_mover_elec = self.e_prime_mover_elec[hour]
        except KeyError:
            E_prime_mover_elec = LpVariable("E_prime_mover_elec{}_hour{}".format(self.name, hour), 0)
            self.e_prime_mover_elec[hour] = E_prime_mover_elec

        return E_prime_mover_elec

    def get_e_prime_mover_elec_aux(self, hour):
        try:
            E_prime_mover_elec_aux = self.e_prime_mover_elec_aux[hour]
        except KeyError:
            E_prime_mover_elec_aux = LpVariable("E_prime_mover_elec{}_hour{}_aux{}".format(self.name, hour, 1), 0, self.xmax_prime_mover - self.xmin_prime_mover)
            self.e_prime_mover_elec_aux[hour] = E_prime_mover_elec_aux

        return E_prime_mover_elec_aux

    def get_constraints(self, hour, xmin_boiler):
        Sturbine = self.get_state_variable(hour)
        E_prime_mover_fuel = self.get_e_prime_mover_fuel(hour, xmin_boiler)
        Q_prime_mover = self.get_q_prime_mover(hour)
        E_prime_mover_elec = self.get_e_prime_mover_elec(hour)
        E_prime_mover_elec_aux = self.get_e_prime_mover_elec_aux(hour)

        constraints = []

        # generator gas
        label = "PrimeMoverFuelConsume{}_{}".format(self.name, hour)
        exp = E_prime_mover_fuel - self.a_E_primer_mover * E_prime_mover_elec_aux - self.b_E_prime_mover * Sturbine == 0
        constraints.append((exp, label))

        # generator heat
        label = "PrimeMoverHeatGenerate{}_{}".format(self.name, hour)
        exp = Q_prime_mover - self.a_Q_primer_mover * E_prime_mover_elec_aux - self.b_Q_primer_mover * Sturbine == 0
        constraints.append((exp, label))

        # microturbine elec
        label = "PrimeMoverElecGenerate{}_{}".format(self.name, hour)
        exp = E_prime_mover_elec - E_prime_mover_elec_aux - self.xmin_prime_mover * Sturbine == 0
        constraints.append((exp, label))

        label = "PrimeMoverElower{}_{}".format(self.name, hour)
        exp = E_prime_mover_elec - self.xmin_prime_mover * Sturbine >= 0
        constraints.append((exp, label))

        label = "PrimeMoverEupper{}_{}".format(self.name, hour)
        exp = E_prime_mover_elec - self.xmax_prime_mover * Sturbine <= 0
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
    prime_movers = []
    for i, (name, parameters) in enumerate(fuel_cell_params.items()):
        mat_prime_mover = parameters["mat_prime_mover"]
        cap_prime_mover = parameters["cap_prime_mover"]

        xmin_prime_mover = cap_prime_mover * 0.3
        a_E_primer_mover = mat_prime_mover[1]
        b_E_prime_mover = mat_prime_mover[0] + a_E_primer_mover * xmin_prime_mover
        xmax_prime_mover = cap_prime_mover

        a_Q_primer_mover = a_E_primer_mover - 1 / 293.1
        b_Q_primer_mover = b_E_prime_mover - xmin_prime_mover / 293.1

        prime_mover = PrimeMover(
            name=name,
            xmin_prime_mover=xmin_prime_mover,
            a_E_primer_mover=a_E_primer_mover,
            b_E_prime_mover=b_E_prime_mover,
            xmax_prime_mover=xmax_prime_mover,
            a_Q_primer_mover=a_Q_primer_mover,
            b_Q_primer_mover=b_Q_primer_mover
        )

        prime_movers.append(prime_mover)


    boilers = []
    for i, (name, parameters) in enumerate(boiler_params.items()):
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

        boiler = Boiler(name=name,
                        xmin_boiler=xmin_boiler,
                        a_boiler=a_boiler,
                        b_boiler=b_boiler,
                        xmax_boiler=xmax_boiler)
        boilers.append(boiler)

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

    absorption_chillers = []
    for i, (name, parameters) in enumerate(absorption_chiller_params.items()):
        mat_abschiller = parameters["mat_abschiller"]
        xmax_abschiller = parameters["xmax_abschiller"]
        xmin_abschiller = parameters["xmin_abschiller"]
        min_on_abs_chiller = parameters.get("min_on_abs_chiller", 3)
        min_off_abs_chiller = parameters.get("min_off_abs_chiller", 0)
        abs_chiller_history = parameters["abs_chiller_history"]
        cap_abs = parameters["cap_abs_chiller"]
        cap_abs = cap_abs / 293.1  # kW -> mmBtu/hr

        xmin_AbsChiller = cap_abs * 0.15
        a_abs = mat_abschiller[1]
        b_abs = mat_abschiller[0] + a_abs * xmin_AbsChiller
        xmax_AbsChiller = cap_abs

        abs_chiller = AbsChiller(name=name,
                                 xmin_abschiller=xmin_AbsChiller,
                                 a_abs=a_abs,
                                 b_abs=b_abs,
                                 xmax_abschiller=xmax_AbsChiller,
                                 history=abs_chiller_history,
                                 min_on_time=min_on_abs_chiller,
                                 min_off_time=min_off_abs_chiller)

        absorption_chillers.append(abs_chiller)


    # heat recovery unit
    a_hru = 0.8

    ################################################################################

    objective_component = []
    constraints = []
    for hour, forecast_hour in enumerate(forecast):
        hour = str(hour).zfill(2)
        
        # free variables
        E_gridelec = LpVariable("E_gridelec_hour{}".format(hour))

        # regular variables
        E_unserve = LpVariable("E_unserve_hour{}".format(hour), 0)
        E_dump = LpVariable("E_dump_hour{}".format(hour), 0)
        Heat_unserve = LpVariable("Heat_unserve_hour{}".format(hour), 0)
        Heat_dump = LpVariable("Heat_dump_hour{}".format(hour), 0)
        Cool_unserve = LpVariable("Cool_unserve_hour{}".format(hour), 0)
        Cool_dump = LpVariable("Cool_dump_hour{}".format(hour), 0)

        Q_HRUheating = LpVariable("Q_HRUheating_hour{}".format(hour), 0)
        Q_Genheating = LpVariable("Q_Genheating_hour{}".format(hour), 0)
        Q_Gencooling = LpVariable("Q_Gencooling_hour{}".format(hour), 0)

        # constraints
        objective_component += [
            forecast_hour["electricity_cost"] * E_gridelec,
            UNSERVE_LIMIT * E_unserve,
            UNSERVE_LIMIT * E_dump,
            UNSERVE_LIMIT * Heat_unserve,
            UNSERVE_LIMIT * Heat_dump,
            UNSERVE_LIMIT * Cool_unserve,
            UNSERVE_LIMIT * Cool_dump
        ]

        objective_component += [forecast_hour["natural_gas_cost"]
                                * b.get_e_boilergas(hour) for b in boilers]

        objective_component += [forecast_hour["natural_gas_cost"]
                                * p.get_e_prime_mover_fuel(hour, xmin_boiler) for p in prime_movers]

        # electric energy balance
        label = "ElecBalance{}".format(hour)
        exp = E_gridelec
        exp += pulp.lpSum([p.get_e_prime_mover_elec(hour) for p in prime_movers])
        for e_chill in [c.get_e_chiller_elec(hour) for c in chillers_igv]:
            exp = exp - e_chill
        exp = exp + E_unserve - E_dump
        exp = exp == forecast_hour["elec_load"] - forecast_hour["solar_kW"]
        constraints.append((exp, label))

        # heating balance
        label = "HeatBalance{}".format(hour)
        exp = pulp.lpSum([b.get_q_boiler(hour) for b in boilers])
        exp = exp + Q_HRUheating + Heat_unserve - Heat_dump == forecast_hour["heat_load"]
        constraints.append((exp, label))

        # cooling balance
        label = "CoolBalance{}".format(hour)
        exp = pulp.lpSum([a.get_q_abs(hour) for a in absorption_chillers] +
                         [c.get_q_chiller(hour) for c in chillers_igv])

        exp = exp + Cool_unserve - Cool_dump
        exp = exp == forecast_hour["cool_load"]
        constraints.append((exp, label))

        # prime_mover
        for prime_mover in prime_movers:
            constraints.extend(prime_mover.get_constraints(hour, xmin_boiler))

        # boiler
        for boiler in boilers:
            constraints.extend(boiler.get_constraints(hour))

        # chillers
        for chiller in chillers_igv:
            constraints.extend(chiller.get_constraints(hour))

        # abschiller
        for abs_chiller in absorption_chillers:
            constraints.extend(abs_chiller.get_constraints(hour, Q_Gencooling))

        # HRU
        label = "HRUWasteheat{}".format(hour)
        exp = Q_Genheating
        exp = exp + Q_Gencooling
        exp = exp - pulp.lpSum([p.get_q_prime_mover(hour) for p in prime_movers])
        exp = exp == 0
        constraints.append((exp, label))

        label = "HRUHeatlimit{}".format(hour)
        exp = Q_HRUheating - a_hru * Q_Genheating <= 0
        constraints.append((exp, label))

    # time lock constraints need to look at multiple state variables
    def lock_on_constraints(label_template, name, min_lock_time, state_variables, state_history):
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

            label = label_template.format(name, hour)
            constraints.append((exp, label))

    def lock_off_constraints(label_template, name, min_lock_time, state_variables, state_history):
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

            label = label_template.format(name, hour)
            constraints.append((exp, label))

    for abs_chiller in absorption_chillers:
        name = abs_chiller.name
        history = abs_chiller.history
        min_on_time = abs_chiller.min_on_time

        state_variables = list(abs_chiller.state_variable.items())
        state_variables = sorted(state_variables, key=lambda x: int(x[0]))
        state_variables = [x[1] for x in state_variables]

        lock_on_constraints("AbsLockOn{}_{}", name, min_on_time,
                            state_variables, history)


    # Build the optimization problem
    prob = pulp.LpProblem("Building Optimization", pulp.LpMinimize)

    objective_function = objective_component[0]
    for component in objective_component[1:]:
        objective_function += component

    prob += objective_function, "Objective Function"

    for c in constraints:
       prob += c

    return prob
