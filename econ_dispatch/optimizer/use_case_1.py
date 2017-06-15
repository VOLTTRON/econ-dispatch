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

def get_optimization_problem(forecast, parameters={}):
    # get the model parameters and bounds for variables
    # load FuelCellPara.mat

    _log.debug("Parameters:\n"+pformat(parameters))

    try:
        mat_turbine = parameters["mat_turbine"]
        xmax_turbine = parameters["xmax_turbine"]
        xmin_turbine = parameters["xmin_turbine"]

        # load BoilerPara.mat
        mat_boiler = parameters["mat_boiler"]
        xmax_boiler =  parameters["xmax_boiler"]
        xmin_boiler =  parameters["xmin_boiler"]

        # load ChillerIGVPara.mat
        mat_chillerIGV = parameters["mat_chillerIGV"]
        xmax_chillerIGV = parameters["xmax_chillerIGV"]
        xmin_chillerIGV = parameters["xmin_chillerIGV"]

        # load AbsChillerPara.mat
        mat_abschiller = parameters["mat_abschiller"]
        xmax_abschiller = parameters["xmax_abschiller"]
        xmin_abschiller = parameters["xmin_abschiller"]
    except KeyError as e:
        raise RuntimeError("Missing needed configuration parameter: " + e.message)


    # component capacity
    cap_FuelCell = 500  # kW
    cap_abs = 464 / 293.1  # kW -> mmBtu/hr
    cap_boiler = 8  # mmBtu/hr
    n_chiller = 3
    cap_chiller = 200 * 3.517 / 293.1  # ton -> mmBtu/hr

    ## compute the parameters for the optimization
    # boiler
    xmin_boiler[0] = cap_boiler * 0.15 # !!!need to consider cases when xmin is not in the first section of the training data
    Nsection = np.where(xmax_boiler > cap_boiler)[0][0]
    Nsection = Nsection + 1
    xmin_boiler = xmin_boiler[:Nsection]
    xmax_boiler = xmax_boiler[:Nsection]
    a_boiler = mat_boiler[1][:Nsection]
    b_boiler = mat_boiler[0][0] + a_boiler[0] * xmin_boiler[0]
    xmax_boiler[-1] = cap_boiler

    # absorption chiller
    flagabs = True
    xmin_AbsChiller = cap_abs * 0.15
    a_abs = mat_abschiller[1]
    b_abs = mat_abschiller[0] + a_abs * xmin_AbsChiller
    xmax_AbsChiller = cap_abs

    # chiller
    xmin_Chiller= []
    a_chiller = []
    b_chiller = []
    xmax_Chiller = []
    for i in range(n_chiller):
        xmin_Chiller.append(cap_chiller * 0.15)
        a_chiller.append(mat_chillerIGV[1] + i * 0.01)  # adding 0.01 to the slopes to differentiate the chillers
        b_chiller.append(mat_chillerIGV[0] + a_chiller[i] * xmin_chillerIGV)
        xmax_Chiller.append(cap_chiller)

    # generator
    xmin_Turbine = cap_FuelCell * 0.3
    a_E_turbine = mat_turbine[1]
    b_E_turbine = mat_turbine[0] + a_E_turbine * xmin_Turbine
    xmax_Turbine = cap_FuelCell

    a_Q_turbine = a_E_turbine - 1 / 293.1
    b_Q_turbine = b_E_turbine - xmin_Turbine / 293.1

    # heat recovery unit
    a_hru = 0.8


    ################################################################################

    objective_component = []
    constraints = []
    for hour, forecast_hour in enumerate(forecast):
        hour = str(hour).zfill(2)
        # binary variables
        Sturbine = binary_var("Sturbine_hour{}".format(hour))
        Sboiler = binary_var("Sboiler_hour{}".format(hour))
        Sabs = binary_var("Sabs_hour{}".format(hour))
        Schiller = []
        for i in range(n_chiller):
            var = binary_var("Schiller{}_hour{}".format(i, hour))
            Schiller.append(var)

        # free variables
        E_gridelec = LpVariable("E_gridelec_hour{}".format(hour))

        # regular variables

        E_unserve = LpVariable("E_unserve_hour{}".format(hour), 0)
        E_dump = LpVariable("E_dump_hour{}".format(hour), 0)
        Heat_unserve = LpVariable("Heat_unserve_hour{}".format(hour), 0)
        Heat_dump = LpVariable("Heat_dump_hour{}".format(hour), 0)
        Cool_unserve = LpVariable("Cool_unserve_hour{}".format(hour), 0)
        Cool_dump = LpVariable("Cool_dump_hour{}".format(hour), 0)

        E_turbinegas = LpVariable("E_turbinegas_hour{}".format(hour), xmin_boiler[0])
        Q_turbine = LpVariable("Q_turbine_hour{}".format(hour), 0)
        E_turbineelec = LpVariable("E_turbineelec_hour{}".format(hour), 0)
        E_turbineelec_aux = LpVariable("E_turbineelec_hour{}_aux{}".format(hour, 1), 0, xmax_Turbine - xmin_Turbine)

        E_boilergas = LpVariable("E_boilergas_hour{}".format(hour), 0)

        Q_boiler = LpVariable("Q_boiler_hour{}".format(hour), 0)
        Q_boiler_aux = []
        for i in range(len(xmax_boiler)):
            var = LpVariable("Q_boiler_hour{}_aux{}".format(hour, i), 0, xmax_boiler[i] - xmin_boiler[i])
            Q_boiler_aux.append(var)

        E_chillerelec = []
        for i in range(n_chiller):
            var = LpVariable("E_chillerelec{}_hour{}".format(i, hour), 0)
            E_chillerelec.append(var)

        Q_chiller = []
        Q_chiller_aux = []
        for i in range(n_chiller):
            var = LpVariable("Q_chiller{}_hour{}".format(i, hour), 0)
            Q_chiller.append(var)

            aux = LpVariable("Q_chiller{}_hour{}_aux{}".format(i, hour, 1), 0, xmax_Chiller[i] - xmin_Chiller[i])
            Q_chiller_aux.append(aux)

        Q_abs = LpVariable("Q_abs_hour{}".format(hour), 0)
        Q_abs_aux = [LpVariable("Q_abs_hour{}_aux0".format(hour), 0)]

        Q_HRUheating = LpVariable("Q_HRUheating_hour{}".format(hour), 0)
        Q_Genheating = LpVariable("Q_Genheating_hour{}".format(hour), 0)
        Q_Gencooling = LpVariable("Q_Gencooling_hour{}".format(hour), 0)

        # constraints
        objective_component += [
            forecast_hour["natural_gas_cost"]* E_turbinegas,
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
        exp = E_turbineelec + E_gridelec
        for e_chill in E_chillerelec:
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
        if flagabs:
            exp = Q_abs
            for q_chill in Q_chiller:
                exp = exp + q_chill
        else:
            exp = Q_chiller[0]
            for q_chill in Q_chiller[1:]:
                exp = exp + q_chill

        exp = exp + Cool_unserve - Cool_dump
        exp = exp == forecast_hour["cool_load"]
        constraints.append((exp, label))

        # generator gas
        label = "TurbineGasConsume{}".format(hour)
        exp = E_turbinegas - a_E_turbine * E_turbineelec_aux - b_E_turbine * Sturbine == 0
        constraints.append((exp, label))

        # generator heat
        label = "TurbineHeatGenerate{}".format(hour)
        exp = Q_turbine - a_Q_turbine * E_turbineelec_aux - b_Q_turbine * Sturbine == 0
        constraints.append((exp, label))

        # microturbine elec
        label = "TurbineElecGenerate{}".format(hour)
        exp = E_turbineelec - E_turbineelec_aux - xmin_Turbine * Sturbine == 0
        constraints.append((exp, label))

        label = "TurbineElower{}".format(hour)
        exp = E_turbineelec - xmin_Turbine * Sturbine >= 0
        constraints.append((exp, label))

        label = "TurbineEupper{}".format(hour)
        exp = E_turbineelec - xmax_Turbine * Sturbine <= 0
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
        for chiller in range(n_chiller):
            label = "ChillerElecConsume{}_{}".format(chiller, hour)
            exp = E_chillerelec[chiller] - a_chiller[chiller] * Q_chiller_aux[chiller]
            # for a_chill, aux in zip(a_chiller, Q_chiller_aux):
            #     exp = exp - a_chill * aux
            exp = exp - b_chiller[chiller] * Schiller[chiller]
            exp = exp == 0
            constraints.append((exp, label))

            label = "ChillerCoolGenerate{}_{}".format(chiller, hour)
            exp = Q_chiller[chiller] - Q_chiller_aux[chiller]
            # for q_chill_aux in Q_chiller_aux:
            #     exp = exp - q_chill_aux
            exp = exp - xmin_Chiller[chiller] * Schiller[chiller]
            exp = exp == 0
            constraints.append((exp, label))

            label = "ChillerQlower{}_{}".format(chiller, hour)
            exp = Q_chiller[chiller] - xmin_Chiller[chiller] * Schiller[chiller] >= 0
            constraints.append((exp, label))

            label = "ChillerQupper{}_{}".format(chiller, hour)
            exp = Q_chiller[chiller] - xmax_Chiller[chiller] * Schiller[chiller] <= 0
            constraints.append((exp, label))


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
        if flagabs:
            exp = exp + Q_Gencooling
        exp = exp - Q_turbine
        exp = exp == 0
        constraints.append((exp, label))

        label = "HRUHeatlimit{}".format(hour)
        exp = Q_HRUheating - a_hru * Q_Genheating <= 0
        constraints.append((exp, label))


    prob = pulp.LpProblem("Building Optimization", pulp.LpMinimize)

    objective_function = objective_component[0]
    for component in objective_component[1:]:
        objective_function += component

    prob += objective_function, "Objective Function"

    for c in constraints:
       prob += c

    return prob
