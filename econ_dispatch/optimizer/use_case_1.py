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


def binary_var(name):
    return LpVariable(name, 0, 1, pulp.LpInteger)

def optimize(forecast, write_lp=False):
    # get the model parameters and bounds for variables
    # load FuelCellPara.mat
    m_Turbine = [0.553388269906111, 0.00770880111175251]
    xmax_Turbine = 408.200000000000
    xmin_Turbine = 20.5100000000000

    # load BoilerPara.mat
    m_Boiler = [[1.02338001783412, -5.47472301330830, -11.2128369305035],
                [1.25837185026029, 1.52937309060521, 1.65693038866453]]
    xmax_Boiler = [23.9781302213668, 44.9845991134643, 61.2098989486694]
    xmin_Boiler = [0.149575993418693, 23.9781302213668, 44.9845991134643]

    # load ChillerIGVPara.mat
    m_ChillerIGV = [14.0040999536939, 35.4182417367909]
    xmax_ChillerIGV = 2.13587853974753
    xmin_ChillerIGV = 0.155991129307404

    # load AbsChillerPara.mat
    m_AbsChiller = [1.42355081496335, 0.426344465964358]
    xmax_AbsChiller = 7.91954964176049
    xmin_AbsChiller = 6.55162743091095


    # component capacity
    cap_FuelCell = 500  # kW
    cap_abs = 464 / 293.1  # kW -> mmBtu/hr
    cap_boiler = 8  # mmBtu/hr
    n_chiller = 4
    cap_chiller = 200 * 3.517 / 293.1  # ton -> mmBtu/hr

    ## compute the parameters for the optimization
    # boiler
    xmin_Boiler[0] = cap_boiler * 0.2 # !!!need to consider cases when xmin is not in the first section of the training data
    Nsection = np.where(xmax_Boiler > cap_boiler)[0][0]
    Nsection = Nsection + 1
    xmin_Boiler = xmin_Boiler[:Nsection]
    xmax_Boiler = xmax_Boiler[:Nsection]
    a_boiler = m_Boiler[1][:Nsection]
    b_boiler = m_Boiler[0][0] + a_boiler[0] * xmin_Boiler[0]
    xmax_Boiler[-1] = cap_boiler

    # absorption chiller
    flagabs = True
    xmin_AbsChiller = cap_abs * 0.2
    a_abs = m_AbsChiller[1]
    b_abs = m_AbsChiller[0] + a_abs * xmin_AbsChiller
    xmax_AbsChiller = cap_abs

    # chiller
    xmin_Chiller= []
    a_chiller = []
    b_chiller = []
    xmax_Chiller = []
    for i in range(n_chiller):
        xmin_Chiller.append(cap_chiller * 0.15)
        a_chiller.append(m_ChillerIGV[1] + i * 0.01)  # adding 0.01 to the slopes to differentiate the chillers
        b_chiller.append(m_ChillerIGV[0] + a_chiller[i] * xmin_ChillerIGV)
        xmax_Chiller.append(cap_chiller)

    # generator
    xmin_Turbine = cap_FuelCell * 0.3
    a_E_turbine = m_Turbine[1]
    b_E_turbine = m_Turbine[0] + a_E_turbine * xmin_Turbine
    xmax_Turbine = cap_FuelCell

    a_Q_turbine = a_E_turbine - 1 / 293.1
    b_Q_turbine = b_E_turbine - xmin_Turbine / 293.1

    # heat recovery unit
    a_hru = 0.8


    ################################################################################



    n_hours = 3
    objective_component = []
    constraints = []
    for hour, forecast_hour in enumerate(forecast):
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
        E_turbinegas = LpVariable("E_turbinegas_hour{}".format(hour), xmin_Boiler[0])
        Q_turbine = LpVariable("Q_turbine_hour{}".format(hour), 0)
        E_turbineelec = LpVariable("E_turbineelec_hour{}".format(hour), 0)
        E_turbineelec_aux = LpVariable("E_turbineelec_hour{}_aux{}".format(hour, 1), 0, xmax_Turbine - xmin_Turbine)

        E_boilergas = LpVariable("E_boilergas_hour{}".format(hour), 0)

        Q_boiler = LpVariable("Q_boiler_hour{}".format(hour), 0)
        Q_boiler_aux = []
        for i in range(len(xmax_Boiler)):
            var = LpVariable("Q_boiler_hour{}_aux{}".format(hour, i), 0, xmax_Boiler[i] - xmin_Boiler[i])
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
            forecast_hour["electricity_cost"] * E_gridelec
        ]

        # electric energy balance
        label = "ElecBalance{}".format(hour)
        exp = E_turbineelec + E_gridelec
        for e_chill in E_chillerelec:
            exp = exp - e_chill
        exp = exp == forecast_hour["elec_load"] - forecast_hour["solar_kW"]
        constraints.append((exp, label))

        # heating balance
        label = "HeatBalance{}".format(hour)
        exp = Q_boiler + Q_HRUheating == forecast_hour["heat_load"]
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

        label = "Eturbinelower{}".format(hour)
        exp = E_turbineelec - xmin_Turbine * Sturbine >= 0
        constraints.append((exp, label))

        label = "Eturbineupper{}".format(hour)
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
        exp = exp - xmin_Boiler[0] * Sboiler
        exp = exp == 0
        constraints.append((exp, label))

        label = "Qboilerlower{}".format(hour)
        exp = Q_boiler - xmin_Boiler[0] * Sboiler >= 0
        constraints.append((exp, label))

        label = "Qboilerupper{}".format(hour)
        exp = Q_boiler - xmax_Boiler[-1] * Sboiler <= 0
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

            label = "Qchillerlower{}_{}".format(chiller, hour)
            exp = Q_chiller[chiller] - xmin_Chiller[chiller] * Schiller[chiller] >= 0
            constraints.append((exp, label))

            label = "Qchillerupper{}_{}".format(chiller, hour)
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

        label = "Qabschillerlower{}".format(hour)
        exp = Q_abs - xmin_AbsChiller * Sabs >= 0
        constraints.append((exp, label))

        label = "Qabschillerupper{}".format(hour)
        exp = Q_abs - xmax_AbsChiller * Sabs <= 0
        constraints.append((exp, label))

        # HRU
        label = "Wasteheat{}".format(label)
        exp = Q_Genheating
        if flagabs:
            exp = exp + Q_Gencooling
        exp = exp - Q_turbine
        exp = exp == 0
        constraints.append((exp, label))

        label = "HRUHeatlimit{}".format(hour)
        exp = Q_Genheating - a_hru * Q_HRUheating <= 0
        constraints.append((exp, label))


    prob = pulp.LpProblem("Building Optimization", pulp.LpMinimize)

    objective_function = objective_component[0]
    for component in objective_component[1:]:
        objective_function += component

    prob += objective_function, "Objective Function"

    #print str(objective_function)
    #print len(constraints)

    #for c in constraints:
    #    print str(c)
    #    prob += c

    if write_lp:
        prob.writeLP("TEST.lp")
    prob.solve()

    status = pulp.LpStatus[prob.status]

    #print status

    result = {}

    for var in prob.variables():
        result[var.name] = var.varValue

    result["Optimization Status"] = status

    return result
