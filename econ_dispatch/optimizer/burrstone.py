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

from collections import OrderedDict
import itertools

import numpy as np
import pytz
import pulp

from econ_dispatch.optimizer import get_pulp_optimization_function


def binary_var(name):
    return pulp.LpVariable(name, 0, 1, pulp.LpInteger)


class BuildAsset(object):
    def __init__(self, fundata={}, component_name=None, min_on=0, min_off=0, capacity=0, maint_cost=0, **kwargs):
        self.fundata = fundata
        for k, v in self.fundata.items():
            if type(v) == dict:
                for k1, v1 in v.items():
                    self.fundata[k][k1] = np.array(v1)
            else:
                self.fundata[k] = v

        self.component_name = component_name

        self.min_on = min_on
        self.min_off = min_off
        self.maint_cost = maint_cost
        self.capacity = capacity
        self.settings = kwargs


class BuildAsset_init(object):
    def __init__(self, status=0, output=0.0, command_history=[0]*24):
        #status1[-1] should always == status if populated.
        self.status = status
        self.status1 = np.array(command_history)
        self.output = output


class Facility(object):
    def __init__(   self, 
                    pmax=None,
                    import_peak_on_startup=None, 
                    generators=[], 
                    boilers=[], 
                    parasite=0.03, 
                    demand_charge=0, 
                    minimum_import=0,
                    component_name=None):
        self.pmax = pmax
        self.import_peak_on_startup = import_peak_on_startup
        self.generators = generators
        self.boilers = boilers
        self.parasite = parasite
        self.demand_charge = demand_charge
        self.minimum_import = minimum_import
        self.component_name = component_name


# variable to indicate that we want all variables that match a pattern
# one item in the tuple key can be RANGE
RANGE = -1

class VariableGroup(object):
    def __init__(self, name, indexes=(), is_binary_var=False, lower_bound_func=None, upper_bound_func=None):
        self.variables = {}

        name_base = name
        for _ in range(len(indexes)):
            name_base += "_{}"

        for index in itertools.product(*indexes):
            var_name = name_base.format(*index)

            if is_binary_var:
                var = binary_var(var_name)
            else:

                # find upper and lower bounds for the variable, if available
                if lower_bound_func is not None:
                    lower_bound = lower_bound_func(index)
                else:
                    lower_bound = None

                if upper_bound_func is not None:
                    upper_bound = upper_bound_func(index)
                else:
                    upper_bound = None

                # the lower bound should always be set if the upper bound is set
                if lower_bound is None and upper_bound is not None:
                    raise RuntimeError("Lower bound should not be unset while upper bound is set")

                # create the lp variable
                if upper_bound is not None:
                    var = pulp.LpVariable(var_name, lower_bound, upper_bound)
                elif lower_bound is not None:
                    var = pulp.LpVariable(var_name, lower_bound)
                else:
                    var = pulp.LpVariable(var_name)

            self.variables[index] = var

    def match(self, key):
        position = key.index(RANGE)
        def predicate(xs, ys):
            z = 0
            for i, (x, y) in enumerate(zip(xs, ys)):
                if i != position and x == y:
                    z += 1
            return z == len(key) - 1


        keys = list(self.variables.keys())
        keys = [k for k in keys if predicate(k, key)]
        keys.sort(key=lambda k: k[position])

        return [self.variables[k] for k in keys]

    def __getitem__(self, key):
        if type(key) != tuple:
            key = (key,)

        # n_range = 0
        # for i, x in enumerate(key):
        #     if x == RANGE:
        #         n_range += 1
        n_range = key.count(RANGE)

        if n_range == 0:
            return self.variables[key]
        elif n_range == 1:
            return self.match(key)
        else:
            raise ValueError("Can only get RANGE for one index.")


def constant(x):
    def _constant(*args, **kwargs):
        return x

    return _constant


def build_problem(forecast, parameters={}):

    # forecast is parasys, parameters are all the other csvs

    parasys = {}
    for fc in forecast:
        for key, value in fc.items():
            try:
                parasys[key].append(value)
            except KeyError:
                parasys[key] = [value]

    # parameters
    params = {
        "cpi": 1,
        "lhv": 905,
        "hhv": 905/0.9,
        "st_hfg": 1019.4,
        "eta_boiler": 0.75,
        "eta_boiler_real": 0.62,
        "st_loss": 0.2,
        "st_kw": 0.005,
        "q_kw": 5.0,
        "timezone_name": "America/New_York"
    }

    # Localize timestamps
    timezone = pytz.timezone(params["timezone_name"])
    try:
        parasys["timestamp"] = [ts.astimezone(timezone) for ts in parasys["timestamp"]]
    except ValueError:
        parasys["timestamp"] = [ts.astimezone(
            pytz.UTC.localize(timezone)) for ts in parasys["timestamp"]]
    # Weekdays between 8:00AM and 8:00PM
    parasys["peak_hours"] = np.array([
        True if (i.weekday() in np.arange(0, 5))
        and (i.hour in np.arange(8, 20))
        else False for i in parasys["timestamp"]])
    # Forecasts are numpy arrays
    for k, v in parasys.iteritems():
        parasys[k] = np.array(v)
    # Could be updated with forecast model
    parasys["cpi"] = parasys.get("cpi", params["cpi"])
    # Depends on cpi and gas price
    parasys["thermal_price"] = 9.421 + 3.918*parasys['cpi']\
        + 0.8396*(parasys['natural_gas_cost'] - 11.22)
    # Depends on thermal price
    parasys["hot_water_price"] = parasys["thermal_price"]/(1-params['st_loss'])
    parasys["steam_price"] = parasys["thermal_price"]*params['st_hfg']/1e6
    # Depends on cpi and gas price
    parasys["college_electricity_price"] = 0.08158 + 0.00998*parasys['cpi']\
        + 0.0073*(parasys['natural_gas_cost'] - 11.22)
    parasys["hospital_electricity_price"] = 0.03214 + 0.01337*parasys['cpi']\
        + 0.0029*(parasys['natural_gas_cost'] - 11.22)
    parasys["home_electricity_price"] = 0.10537 - 0.00293*parasys['cpi']\
        + 0.0094*(parasys['natural_gas_cost'] - 11.22)
    # Update gas price units
    parasys['natural_gas_cost'] = parasys['natural_gas_cost']*params['hhv']/1e6

    # map import cost categories to facilities
    parasys["college_import_cost"] = parasys.get("college_import_cost", 
                                                 parasys["utica_sc3_primary"])
    parasys["hospital_import_cost"] = parasys.get("hospital_import_cost", 
                                                  parasys["utica_sc3a_primary"])
    parasys["home_import_cost"] = parasys.get("home_import_cost", 
                                              parasys["utica_sc3_secondary"])

    # print "================================================================================"
    # for k, v in parasys.items():
    #     print k
    # print "================================================================================"
    # import json
    # print(json.dumps(parameters, indent=4, sort_keys=True))
    # print "================================================================================"
    # import json
    # _temp = {}
    # for k, v in parasys.iteritems():
    #     try:
    #         _temp[k] = [str(vv) for vv in v]
    #     except TypeError:
    #         _temp[k] = str(v)
    # with open("parasys.json", "w") as f:
    #     json.dump(_temp, f, indent=4, sort_keys=True)
    # with open("parameters.json", "w") as f:
    #     json.dump(parameters, f, indent=4, sort_keys=True)

    generator_params = parameters.get("burrstone_generator", {})
    boiler_params = parameters.get("burrstone_boiler", {})
    facility_params = parameters.get("burrstone_facility", {})

    generator_para = OrderedDict()
    generator_init = OrderedDict()
    for name, parameters in generator_params.items():
        fundata = parameters["fundata"]
        min_on = parameters["min_on"]
        min_off = parameters["min_off"]
        maint_cost = parameters["maint_cost"]
        command_history = parameters["command_history"]
        capacity = parameters["capacity"]
        generator_para[name] = BuildAsset(component_name=name,
                                        fundata=fundata,
                                        min_on=min_on,
                                        min_off=min_off,
                                        capacity=capacity,
                                        maint_cost=maint_cost)
        generator_init[name] = BuildAsset_init(status=command_history[-1], command_history=command_history)

    boiler_para = OrderedDict()
    for name, parameters in boiler_params.items():
        fundata = parameters["fundata"]
        boiler_para[name] = BuildAsset(component_name=name, fundata=fundata)

    facility_para = OrderedDict()
    facility_init = OrderedDict()
    for name, parameters in facility_params.items():
        pmax = parameters['pmax']
        import_peak_on_startup = parameters['import_peak_on_startup']
        generators = parameters['generators']
        boilers = parameters['boilers']
        parasite = parameters['parasite']
        demand_charge = parameters['demand_charge']
        minimum_import = parameters['minimum_import']
        facility_para[name] = Facility(pmax=pmax, 
                                    import_peak_on_startup=import_peak_on_startup, 
                                    generators=generators,
                                    boilers=boilers,
                                    parasite=parasite,
                                    demand_charge=demand_charge,
                                    minimum_import=minimum_import,
                                    component_name=name)

    generator_names = tuple([x.component_name for x in generator_para.values()])
    boiler_names = tuple([x.component_name for x in boiler_para.values()])
    facility_names = tuple([x.component_name for x in facility_para.values()])


    # constants
    H_t = len(parasys["timestamp"])
    KK = 2
    BIGM = 10000
    constant_zero = constant(0)


    # variables
    index_generator = generator_names, range(H_t)
    index_generator_kk = generator_names, range(KK), range(H_t)
    generator_elec = VariableGroup("generator_elec", indexes=index_generator, lower_bound_func=constant_zero)
    generator_s = VariableGroup("generator_s", indexes=index_generator, is_binary_var=True)
    generator_vars = {
        "gas": {
            "y": VariableGroup("generator_gas", indexes=index_generator, lower_bound_func=constant_zero),
            "x_k": VariableGroup("generator_gas_x_k", indexes=index_generator_kk, lower_bound_func=constant_zero),
            "x_s_k": VariableGroup("generator_gas_x_s_k", indexes=index_generator_kk, is_binary_var=True)
        },
        "st": {
            "y": VariableGroup("generator_st", indexes=index_generator, lower_bound_func=constant_zero),
            "x_k": VariableGroup("generator_st_k", indexes=index_generator_kk, lower_bound_func=constant_zero),
            "x_s_k": VariableGroup("generator_st_s_k", indexes=index_generator_kk, is_binary_var=True)
        },
        "q": {
            "y": VariableGroup("generator_q", indexes=index_generator, lower_bound_func=constant_zero),
            "x_k": VariableGroup("generator_q_k", indexes=index_generator_kk, lower_bound_func=constant_zero),
            "x_s_k": VariableGroup("generator_q_s_k", indexes=index_generator_kk, is_binary_var=True)
        }
    }

    index_boiler = boiler_names, range(H_t)
    boiler_on = VariableGroup("boiler_on", indexes=index_boiler, is_binary_var=True)
    boiler_st = VariableGroup("boiler_st", indexes=index_boiler, lower_bound_func=constant_zero)
    boiler_gas = VariableGroup("boiler_gas", indexes=index_boiler, lower_bound_func=constant_zero)

    index_facility = facility_names, range(H_t)
    index_facility_no_hour = (facility_names,)
    import_peak = VariableGroup("import_peak", indexes=(facility_names,), lower_bound_func=constant_zero)
    facility_ex = VariableGroup("facility_ex", indexes=index_facility, lower_bound_func=constant_zero)
    facility_im = VariableGroup("facility_im", indexes=index_facility, lower_bound_func=constant_zero)
    facility_b = VariableGroup("facility_b", indexes=index_facility, is_binary_var=True)

    E_DUMP = VariableGroup("E_DUMP", indexes=index_facility, lower_bound_func=constant_zero)
    E_UNSERVE = VariableGroup("E_UNSERVE", indexes=index_facility, lower_bound_func=constant_zero)

    index_hour = (range(H_t),)
    reject_st = VariableGroup("reject_st", indexes=index_hour, lower_bound_func=constant_zero)
    reject_q = VariableGroup("reject_q", indexes=index_hour, lower_bound_func=constant_zero)
    qmin = VariableGroup("qmin", indexes=index_hour, lower_bound_func=constant_zero)

    Q_DUMP = VariableGroup("Q_DUMP", indexes=index_hour, lower_bound_func=constant_zero)
    Q_UNSERVE = VariableGroup("Q_UNSERVE", indexes=index_hour, lower_bound_func=constant_zero)
    ST_DUMP = VariableGroup("ST_DUMP", indexes=index_hour, lower_bound_func=constant_zero)
    ST_UNSERVE = VariableGroup("ST_UNSERVE", indexes=index_hour, lower_bound_func=constant_zero)

    # CONSTRAINTS
    constraints = []

    def add_constraint(name, indexes, constraint_func):
        name_base = name
        for _ in range(len(indexes)):
            name_base += "_{}"

        for index in itertools.product(*indexes):
            name = name_base.format(*index)
            c = constraint_func(index)

            constraints.append((c, name))


    # Generator consume
    index_KK = (range(KK),)
    for var, gen_var in generator_vars.items():
        for i, t in itertools.product(*index_generator):
            name = "generator_{}_{}_{}".format(i, var, t)
            partial = []
            for k in index_KK[0]:
                partial.append(generator_para[i].fundata[var]["a"][k]*gen_var["x_k"][i,k,t])
                partial.append(generator_para[i].fundata[var]["b"][k]*gen_var["x_s_k"][i,k,t])
            c = gen_var["y"][i,t] == pulp.lpSum(partial)
            constraints.append((c, name))

            name = "generator_{}_elec_{}_{}".format(i, var, t)
            c = generator_elec[i,t] == pulp.lpSum(gen_var["x_k"][i,RANGE,t])
            constraints.append((c, name))

            name = "generator_{}_{}_x_{}_status".format(i, var, t)
            c = generator_s[i,t] == pulp.lpSum(gen_var["x_s_k"][i,RANGE,t])
            constraints.append((c, name))
            
            for k in index_KK[0]:
                name = "generator_{}_{}_x_lower_{}_{}".format(i, var, k, t)
                c = generator_para[i].fundata[var]["min"][k] * gen_var["x_s_k"][i,k,t] <= gen_var["x_k"][i,k,t]
                constraints.append((c, name))
                
                name = "generator_{}_{}_x_upper_{}_{}".format(i, var, k, t)
                c = generator_para[i].fundata[var]["max"][k] * gen_var["x_s_k"][i,k,t] >= gen_var["x_k"][i,k,t]
                constraints.append((c, name))

    # Generator min on
    for i in generator_names:
        for t in range(0,1):
            name = "generatorlockon_{}_{}".format(i, t)
            c = generator_para[i].min_on * (generator_init[i].status1[-1] - generator_s[i,t]) <= pulp.lpSum(generator_init[i].status1[H_t+t-generator_para[i].min_on:H_t])
            constraints.append((c, name))

        for t in range(1, generator_para[i].min_on):
            name = "generatorlockon_{}_{}".format(i, t)
            partial = []
            for tau in range(0, t):
                partial.append(generator_s[i,tau])
            c = generator_para[i].min_on * (generator_s[i,t-1] - generator_s[i,t]) <= pulp.lpSum(partial) + pulp.lpSum(generator_init[i].status1[H_t+t-generator_para[i].min_on:H_t])
            constraints.append((c, name))
            
        for t in range(generator_para[i].min_on, H_t):
            name = "generatorlockon_{}_{}".format(i, t)
            partial = []
            for tau in range(t-generator_para[i].min_on, t):
                partial.append(generator_s[i,tau])
            c = generator_para[i].min_on * (generator_s[i,t-1] - generator_s[i,t]) <= pulp.lpSum(partial)
            constraints.append((c, name))

    # Generator min off
    for i in generator_names:
        for t in range(0,1):
            name = "generatorlockoff_{}_{}".format(i, t)
            c = generator_para[i].min_off * (generator_s[i,t] - generator_init[i].status1[-1]) <= pulp.lpSum(1 - generator_init[i].status1[H_t+t-generator_para[i].min_off:H_t])
            constraints.append((c, name))

        for t in range(1, generator_para[i].min_off):
            name = "generatorlockoff_{}_{}".format(i, t)
            partial = []
            for tau in range(0, t):
                partial.append(1 - generator_s[i,tau])
            c = generator_para[i].min_off * (generator_s[i,t] - generator_s[i,t-1]) <= pulp.lpSum(partial) + pulp.lpSum(1 - generator_init[i].status1[H_t+t-generator_para[i].min_off:H_t])
            constraints.append((c, name))
            
        for t in range(generator_para[i].min_off, H_t):
            name = "generatorlockoff_{}_{}".format(i, t)
            partial = []
            for tau in range(t-generator_para[i].min_off, t):
                partial.append(1 - generator_s[i,tau])
            c = generator_para[i].min_off * (generator_s[i,t] - generator_s[i,t-1]) <= pulp.lpSum(partial)
            constraints.append((c, name))

    # Prefer hospital0 to hospital1 to speed convergence
    for t in index_hour[0]:
        # [t=1:H_t]
        name = "prefer_generator0_{}".format(t)
        c = generator_s['hospital0',t] >= generator_s['hospital1',t]
        constraints.append((c,name))


    # Boiler
    def boilerxGenerate(index):
        # [i=1:N_boiler,t=1:H_t]
        i, t = index
        return boiler_gas[i,t] == boiler_st[i,t]*boiler_para[i].fundata['eff']
    add_constraint("boilerxGenerate", index_boiler, boilerxGenerate)

    # Boiler cannot overproduce
    def boilercapacity(index):
        # [i=1:N_boiler,t=1:H_t]
        i, t = index
        return boiler_st[i,t] <= boiler_on[i,t]*parasys['steam_load'][t]
    add_constraint("boilercapacity", index_boiler, boilercapacity)

    # Require all boilers to be off before rejecting steam
    def rejectsteam(index):
        # [t=1:H_t]
        t = index[0]
        return reject_st[t] <= pulp.lpSum([1 - b for b in boiler_on[RANGE,t]])*BIGM
    add_constraint("rejectsteam", index_hour, rejectsteam)


    # Facility
    # NOTE: if minimum_import > elec_load for any facility at any time, the problem is infeasible

    def facilityimport(index):
        # [i=1:N_facility,t=1:H_t]
        i, t = index
        return facility_im[i,t] <= facility_b[i,t]*parasys['{}_electricity_load'.format(i)][t]
    add_constraint("facilityimport", index_facility, facilityimport)

    # Minimum import 
    def facilityminimumimport(index):
        # [i=1:N_facility,t=1:H_t]
        i, t = index
        return facility_im[i,t]>=facility_para[i].minimum_import
    add_constraint("facilityminimumimport", index_facility, facilityminimumimport)

    def facilityexport(index):
        # [i=1:N_facility,t=1:H_t]
        i, t = index
        return facility_ex[i,t] <= (1-facility_b[i,t])*facility_para[i].pmax
    add_constraint("facilityexport", index_facility, facilityexport)

    def facilitydemand0(index):
        # [i=1:N_facility]
        i = index[0]
        return import_peak[i] >= facility_para[i].import_peak_on_startup
    add_constraint("facilitydemand0", index_facility_no_hour, facilitydemand0)

    def facilitydemand1(index):
        # [i=1:N_facility,t=1:H_t]
        i, t = index
        return import_peak[i] >= facility_im[i,t]*parasys['peak_hours'][t]
    add_constraint("facilitydemand1", index_facility, facilitydemand1)

    for t in index_hour[0]:
        # [t=1:H_t]
        name = "home_import_constraint_{}".format(t)
        c = facility_b['home',t] == 1
        constraints.append((c, name))

    # Can't sell reject_q: charge min(hot_water_load, generated_q) instead.
    # Since hot_water_load <= generated_q, should just be qmin == hot_water_load
    def QMinDemand(index):
        # [t=1:H_t]
        t = index[0]
        return qmin[t] <= parasys["hot_water_load"][t]
    add_constraint("QMinDemand", index_hour, QMinDemand)

    def QMinGenerate(index):
        # [t=1:H_t]
        t = index[0]
        return qmin[t] <= pulp.lpSum(generator_vars['q']['y'][RANGE,t])
    add_constraint("QMinGenerate", index_hour, QMinGenerate)


    # Balances
    def HeatBalance(index):
        # [t=1:H_t]
        t = index[0]
        return pulp.lpSum([generator_vars['q']['y'][i,t] for i in generator_names])\
                - reject_q[t]\
                == parasys["hot_water_load"][t]
    add_constraint("HeatBalance", index_hour, HeatBalance)
                # - Q_DUMP[t]\
                # + Q_UNSERVE[t]\

    def SteamBalance(index):
        # [t=1:H_t]
        t = index[0]
        return pulp.lpSum([generator_vars['st']['y'][i,t] for i in generator_names])\
                + pulp.lpSum([boiler_st[i,t] for i in boiler_names])\
                - reject_st[t]\
                == parasys["steam_load"][t]
    add_constraint("SteamBalance", index_hour, SteamBalance)
                # - ST_DUMP[t]\
                # + ST_UNSERVE[t]\

    def ElecBalance(index):
        # [i=1:N_facility,t=1:H_t]
        i, t = index
        return pulp.lpSum([generator_elec[j,t] for j in facility_para[i].generators])\
            + facility_im[i,t]\
            - facility_ex[i,t]\
            - pulp.lpSum([facility_para[i].parasite*generator_para[j].capacity*generator_s[j,t] for j in facility_para[i].generators])\
            == parasys['{}_electricity_load'.format(i)][t]
    add_constraint("ElecBalance", index_facility, ElecBalance)
            #    - E_DUMP[i,t]\
            #    + E_UNSERVE[i,t]\


    # Sales objective variables
    electric_sales = VariableGroup("electric_sales", indexes=index_facility)
    def ElecSales(index):
        i, t = index
        return electric_sales[i,t] == -1.*parasys['{}_electricity_load'.format(i)][t]*parasys['{}_electricity_price'.format(i)][t]
    add_constraint("ElecSales", index_facility, ElecSales)

    generator_steam_sales = VariableGroup("generator_steam_sales", indexes=index_generator)
    def GenSteamSales(index):
        i, t = index
        return generator_steam_sales[i,t] == -1.*generator_vars['st']['y'][i,t]*parasys['steam_price'][t]/params['eta_boiler']
    add_constraint("GenSteamSales", index_generator, GenSteamSales)

    boiler_steam_sales = VariableGroup("boiler_steam_sales", indexes=index_boiler)
    def BoilerSteamSales(index):
        i, t = index
        return boiler_steam_sales[i,t] == -1.*boiler_st[i,t]*parasys['steam_price'][t]
    add_constraint("BoilerSteamSales", index_boiler, BoilerSteamSales)

    generator_heat_sales = VariableGroup("generator_heat_sales", indexes=index_hour)
    def GenHeatSales(index):
        t = index[0]
        return generator_heat_sales[t] == -1.*qmin[t]*parasys['hot_water_price'][t]/params['eta_boiler']
    add_constraint("GenHeatSales", index_hour, GenHeatSales)

    electric_export_sales = VariableGroup("electric_export_sales", indexes=index_facility)
    def ElectricExportSales(index):
        i,t = index
        return electric_export_sales[i,t] == -1.*facility_ex[i,t]*parasys['electricity_export_price'][t]
    add_constraint("ElectricExportSales", index_facility, ElectricExportSales)

    # Cost objective variables
    electric_import_costs = VariableGroup("electric_import_costs", indexes=index_facility)
    def ElectricImportCosts(index):
        i,t = index
        return electric_import_costs[i,t] == facility_im[i,t]*parasys['{}_import_cost'.format(i)][t]
    add_constraint("ElectricImportCosts", index_facility, ElectricImportCosts)

    demand_charge_costs = VariableGroup("demand_charge_costs", indexes=index_facility_no_hour, lower_bound_func=constant_zero)
    def DemandChargeCosts(index):
        i = index[0]
        return demand_charge_costs[i] == import_peak[i]*facility_para[i].demand_charge
    add_constraint("DemandChargeCosts", index_facility_no_hour, DemandChargeCosts)

    generator_fuel_costs = VariableGroup("generator_fuel_costs", indexes=index_generator)
    def GeneratorFuelCosts(index):
        i,t = index
        return generator_fuel_costs[i,t] == generator_vars['gas']['y'][i,t]*parasys['natural_gas_cost'][t]
    add_constraint("GeneratorFuelCosts", index_generator, GeneratorFuelCosts)

    boiler_fuel_costs = VariableGroup("boiler_fuel_costs", indexes=index_boiler)
    def BoilerFuelCosts(index):
        i,t = index
        return boiler_fuel_costs[i,t] == boiler_gas[i,t]*parasys['natural_gas_cost'][t]
    add_constraint("BoilerFuelCosts", index_boiler, BoilerFuelCosts)

    maintenance_costs = VariableGroup("maintenance_cost", indexes=index_generator)
    def MaintenanceCosts(index):
        i,t = index
        return maintenance_costs[i,t] == generator_s[i,t]*generator_para[i].maint_cost
    add_constraint("MaintenanceCosts", index_generator, MaintenanceCosts)

    heat_rejection_costs = VariableGroup("heat_rejection_costs", indexes=index_hour)
    def HeatRejectionCost(index):
        # Based on electricity price at hospital
        t = index[0]
        return heat_rejection_costs[t] == (params['q_kw']*reject_q[t]+params['st_kw']*reject_st[t])*parasys['hospital_electricity_price'][t]
    add_constraint("HeatRejectionCost", index_hour, HeatRejectionCost)


    # Objective function
    objective_components = []
    for t in range(H_t):
        for i in index_facility[0]:
            # objective_components.append(electric_import_costs[i,t])
            objective_components.append(demand_charge_costs[i])
            objective_components.append(electric_sales[i,t])
            objective_components.append(electric_export_sales[i,t])
            # objective_components.append(BIGM*E_UNSERVE[i,t])
            # objective_components.append(BIGM*E_DUMP[i,t])

        for i in index_generator[0]:
            objective_components.append(generator_fuel_costs[i,t])
            objective_components.append(maintenance_costs[i,t])
            objective_components.append(generator_steam_sales[i,t])

        for i in index_boiler[0]:
            objective_components.append(boiler_fuel_costs[i,t])
            objective_components.append(boiler_steam_sales[i,t])
        
        objective_components.append(heat_rejection_costs[t])
        objective_components.append(generator_heat_sales[t])

        # for var in [Q_DUMP, Q_UNSERVE, ST_DUMP, ST_UNSERVE]:
        #     objective_components.append(BIGM*var[t])

    prob = pulp.LpProblem("Building Optimization", pulp.LpMinimize)
    prob += pulp.lpSum(objective_components), "Objective Function"

    for c in constraints:
        prob += c

    return prob


def get_optimization_function(config):
    return get_pulp_optimization_function(build_problem, config)
    