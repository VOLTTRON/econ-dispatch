from collections import OrderedDict
import itertools

import numpy as np
import pulp


def binary_var(name):
    return pulp.LpVariable(name, 0, 1, pulp.LpInteger)


class BuildAsset(object):
    def __init__(self, fundata=None, component_name=None, ramp_up=None, ramp_down=None, startcost=0, min_on=0, min_off=0):
        self.fundata = {}
        for k, v in fundata.items():
            self.fundata[k] = np.array(v)

        self.component_name = component_name

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.startcost = startcost
        self.min_on = min_on
        self.min_off = min_off


class BuildAsset_init(object):
    def __init__(self, status=0, output=0.0):
        self.status = status
        self.status1 = np.zeros(24)
        self.output = output


class Storage(object):
    def __init__(self, pmax, Emax, eta_ch, eta_disch, soc_max=1.0, soc_min=0.0):
        self.pmax = pmax
        self.Emax = Emax
        self.eta_ch = eta_ch
        self.eta_disch = eta_disch
        self.soc_max = soc_max
        self.soc_min = soc_min


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

        n_range = 0
        for i, x in enumerate(key):
            if x == RANGE:
                n_range += 1

        if n_range == 0:
            return self.variables[key]
        elif n_range == 1:
            return self.match(key)
        else:
            raise ValueError("Can only get RANGE for one index.")


def get_optimization_problem(forecast, parameters={}):

    # forecast is parasys, parameters are all the other csvs

    parasys = {}
    for fc in forecast:
        for key, value in fc.items():
            try:
                parasys[key].append(value)
            except KeyError:
                parasys[key] = [value]

    print "================================================================================"
    for k, v in parasys.items():
        print k

    print "================================================================================"
    for k, v in parameters.items():
        print k,v
    print "================================================================================"

    fuel_cell_params = parameters["fuel_cell"]
    microturbine_params = parameters["micro_turbine_generator"]
    boiler_params = parameters["boiler"]
    chiller_params = parameters["centrifugal_chiller_igv"]
    abs_params = parameters["absorption_chiller"]

    turbine_para = OrderedDict()
    turbine_init = OrderedDict()
    for name, parameters in itertools.chain(fuel_cell_params.items(), microturbine_params.items()):
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        startcost = parameters["startcost"]
        min_on = parameters["min_on"]
        output = parameters["output"]
        turbine_para[name] = BuildAsset(fundata=fundata, component_name=name, ramp_up=ramp_up, ramp_down=ramp_down, startcost=startcost) # fuel cell
        turbine_init[name] = BuildAsset_init(status=1, output=output)

    # turbine_para.append(BuildAsset(fundata=pandas.read_csv("paraturbine1.csv"), ramp_up=150, ramp_down=-150, startcost=20, min_on=3)) # fuel cell
    # turbine_para.append(BuildAsset(fundata=pandas.read_csv("paraturbine2.csv"), ramp_up=200, ramp_down=-200, startcost=10, min_on=3)) # microturbine
    # turbine_para.append(BuildAsset(fundata=pandas.read_csv("paraturbine3.csv"), ramp_up=20, ramp_down=-20, startcost=5, min_on=3)) # diesel


    # turbine_init.append(BuildAsset_init(status=1, output=300.0))
    # turbine_init.append(BuildAsset_init(status=1, output=250.0))
    # turbine_init.append(BuildAsset_init(status=0))
    # turbine_init[0].status1[21:] = 1
    # turbine_init[1].status1[21:] = 1

    boiler_para = OrderedDict()
    boiler_init = OrderedDict()
    for name, parameters in boiler_params.items():
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        startcost = parameters["startcost"]
        boiler_para[name] = BuildAsset(fundata=fundata, component_name=name, ramp_up=ramp_up, ramp_down=ramp_down, startcost=startcost)
        boiler_init[name] = BuildAsset_init(status=0)

    # boiler_para.append(BuildAsset(fundata=pandas.read_csv("paraboiler1.csv"), ramp_up=8, ramp_down=-8, startcost=0.8)) # boiler1
    # boiler_para.append(BuildAsset(fundata=pandas.read_csv("paraboiler2.csv"), ramp_up=2, ramp_down=-2, startcost=0.25)) # boiler2
    # boiler_init.append(BuildAsset_init(status=0))
    # boiler_init.append(BuildAsset_init(status=0))

    chiller_para = OrderedDict()
    chiller_init = OrderedDict()
    for name, parameters, in chiller_params.items():
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        startcost = parameters["startcost"]
        chiller_para[name] = BuildAsset(fundata=fundata, component_name=name, ramp_up=ramp_up, ramp_down=ramp_down, startcost=startcost)
        chiller_init[name] = BuildAsset_init(status=0)

    # chiller_para.append(BuildAsset(fundata=pandas.read_csv("parachiller.csv"), ramp_up=6, ramp_down=-6, startcost=15)) # chiller1
    # temp_data = pandas.read_csv("parachiller.csv")
    # temp_data["a"] += +1e-4
    # temp_data["b"] += +1e-4
    # chiller_para.append(BuildAsset(fundata=temp_data, ramp_up=1.5, ramp_down=-1.5, startcost=20)) # chiller2
    # chiller_para.append(BuildAsset(fundata=pandas.read_csv("parachiller3.csv"), ramp_up=1, ramp_down=-1, startcost=5)) # chiller3

    # chiller_init.append(BuildAsset_init(status=0))
    # chiller_init.append(BuildAsset_init(status=1, output=1))
    # chiller_init[2].status1[-1]=1


    abs_para = OrderedDict()
    abs_init = OrderedDict()
    for name, parameters, in abs_params.items():
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        startcost = parameters["startcost"]
        abs_para[name] = BuildAsset(fundata=fundata, component_name=name, ramp_up=ramp_up, ramp_down=ramp_down, startcost=startcost)# chiller1
        abs_init[name] = BuildAsset_init(status=0)

    # abs_para.append(BuildAsset(fundata=pandas.read_csv("paraabs.csv"), ramp_up=0.25, ramp_down=-0.25, startcost=2))# chiller1
    # abs_init.append(BuildAsset_init(status=0))

    KK=3 # number of pieces in piecewise model

    a_hru=0.8

    E_storage_para = []
    E_storage_para.append(Storage(Emax=2000.0, pmax=500.0, eta_ch=0.93, eta_disch=0.97, soc_min=0.1))

    Cool_storage_para = []
    Cool_storage_para.append(Storage(Emax=20.0, pmax=5.0, eta_ch=0.94, eta_disch=0.94))

    bigM = 1e4
    H_t = len(parasys["electricity_cost"])

    # don't like these
    turbine_names = tuple([x.component_name for x in turbine_para.values()])
    # N_turbine = len(turbine_para)

    boiler_names = tuple([x.component_name for x in boiler_para.values()])
    # N_boiler = len(boiler_para)

    chiller_names = tuple([x.component_name for x in chiller_para.values()])
    # N_chiller = len(chiller_para)

    abs_names = tuple([x.component_name for x in abs_para.values()])
    # N_abs = len(abs_para)

    N_E_storage = len(E_storage_para)
    N_Cool_storage = len(Cool_storage_para)

    def constant_zero(*args, **kwargs):
        return 0

    index_turbine = turbine_names, range(H_t)
    turbine_y = VariableGroup("turbine_y", indexes=index_turbine, lower_bound_func=constant_zero)
    turbine_x = VariableGroup("turbine_x", indexes=index_turbine, lower_bound_func=constant_zero)
    turbine_x_k = VariableGroup("turbine_x_k", indexes=index_turbine + (range(KK),), lower_bound_func=constant_zero)
    turbine_s = VariableGroup("turbine_s", indexes=index_turbine, is_binary_var=True)
    turbine_s_k = VariableGroup("turbine_s_k", indexes=index_turbine + (range(KK),), is_binary_var=True)
    turbine_start = VariableGroup("turbine_start", indexes=index_turbine, is_binary_var=True)

    index_boiler = boiler_names, range(H_t)
    boiler_y = VariableGroup("boiler_y", indexes=index_boiler, lower_bound_func=constant_zero)
    boiler_x = VariableGroup("boiler_x", indexes=index_boiler, lower_bound_func=constant_zero)
    boiler_x_k = VariableGroup("boiler_x_k", indexes=index_boiler + (range(KK),), lower_bound_func=constant_zero)
    boiler_s = VariableGroup("boiler_s", indexes=index_boiler, is_binary_var=True)
    boiler_s_k = VariableGroup("boiler_s_k", indexes=index_boiler + (range(KK),), is_binary_var=True)
    boiler_start = VariableGroup("boiler_start", indexes=index_boiler, is_binary_var=True)

    index_chiller = chiller_names, range(H_t)
    chiller_y = VariableGroup("chiller_y", indexes=index_chiller, lower_bound_func=constant_zero)
    chiller_x = VariableGroup("chiller_x", indexes=index_chiller, lower_bound_func=constant_zero)
    chiller_x_k = VariableGroup("chiller_x_k", indexes=index_chiller + (range(KK),), lower_bound_func=constant_zero)
    chiller_s = VariableGroup("chiller_s", indexes=index_chiller, is_binary_var=True)
    chiller_s_k = VariableGroup("chiller_s_k", indexes=index_chiller + (range(KK),), is_binary_var=True)
    chiller_start = VariableGroup("chiller_start", indexes=index_chiller, is_binary_var=True)

    index_abs = abs_names, range(H_t)
    abs_y = VariableGroup("abs_y", indexes=index_abs, lower_bound_func=constant_zero)
    abs_x = VariableGroup("abs_x", indexes=index_abs, lower_bound_func=constant_zero)
    abs_x_k = VariableGroup("abs_x_k", indexes=index_abs + (range(KK),), lower_bound_func=constant_zero)
    abs_s = VariableGroup("abs_s", indexes=index_abs, is_binary_var=True)
    abs_s_k = VariableGroup("abs_s_k", indexes=index_abs + (range(KK),), is_binary_var=True)
    abs_start = VariableGroup("abs_start", indexes=index_abs, is_binary_var=True)


    # the storage variables start to get more complicated
    def e_storage_pmax(index):
        i, t = index
        return E_storage_para[i].pmax

    index_e_storage = range(N_E_storage), range(H_t)
    E_storage_disch = VariableGroup("E_storage_disch",
                                    indexes=index_e_storage,
                                    lower_bound_func=constant_zero,
                                    upper_bound_func=e_storage_pmax)

    E_storage_ch = VariableGroup("E_storage_ch",
                                 indexes=index_e_storage,
                                 lower_bound_func=constant_zero,
                                 upper_bound_func=e_storage_pmax)

    def e_storage_state_lower_bound(index):
        i, t = index
        return E_storage_para[i].Emax * E_storage_para[i].soc_min

    def e_storage_state_upper_bound(index):
        i, t = index
        return E_storage_para[i].Emax * E_storage_para[i].soc_max

    E_storage_state = VariableGroup("E_storage_state",
                                    indexes=index_e_storage,
                                    lower_bound_func=e_storage_state_lower_bound,
                                    upper_bound_func=e_storage_state_upper_bound)


    def cool_storage_pmax(index):
        i, t = index
        return Cool_storage_para[i].pmax

    index_cool_storage = range(N_Cool_storage), range(H_t)
    Cool_storage_disch = VariableGroup("Cool_storage_disch",
                                       indexes=index_cool_storage,
                                       lower_bound_func=constant_zero,
                                       upper_bound_func=cool_storage_pmax)

    Cool_storage_ch = VariableGroup("Cool_storage_ch",
                                    indexes=index_cool_storage,
                                    lower_bound_func=constant_zero,
                                    upper_bound_func=cool_storage_pmax)

    def cool_storage_state_lower_bound(index):
        i, t = index
        return Cool_storage_para[i].Emax * Cool_storage_para[i].soc_min

    def cool_storage_state_upper_bound(index):
        i, t = index
        return Cool_storage_para[i].Emax * Cool_storage_para[i].soc_max

    Cool_storage_state = VariableGroup("Cool_storage_state",
                                       indexes=index_cool_storage,
                                       lower_bound_func=cool_storage_state_lower_bound,
                                        upper_bound_func=cool_storage_state_upper_bound)


    # these are pretty simple
    index_hour = (range(H_t),)
    E_gridelecfromgrid = VariableGroup("E_gridelecfromgrid", indexes=index_hour, lower_bound_func=constant_zero)
    E_gridelectogrid = VariableGroup("E_gridelectogrid", indexes=index_hour, lower_bound_func=constant_zero)
    Q_HRUheating_in = VariableGroup("Q_HRUheating_in", indexes=index_hour, lower_bound_func=constant_zero)
    Q_HRUheating_out = VariableGroup("Q_HRUheating_out", indexes=index_hour, lower_bound_func=constant_zero)

    E_unserve = VariableGroup("E_unserve", indexes=index_hour, lower_bound_func=constant_zero)
    E_dump = VariableGroup("E_dump", indexes=index_hour, lower_bound_func=constant_zero)
    Heat_unserve = VariableGroup("Heat_unserve", indexes=index_hour, lower_bound_func=constant_zero)
    Heat_dump = VariableGroup("Heat_dump", indexes=index_hour, lower_bound_func=constant_zero)
    Cool_unserve = VariableGroup("Cool_unserve", indexes=index_hour, lower_bound_func=constant_zero)
    Cool_dump = VariableGroup("Cool_dump", indexes=index_hour, lower_bound_func=constant_zero)

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


    def ElecBalance(index):
        # [t=1:H_t]
        t = index[0]
        return pulp.lpSum(turbine_x[RANGE,t])\
            + E_gridelecfromgrid[t]\
            - E_gridelectogrid[t]\
            - pulp.lpSum(chiller_y[RANGE,t])\
            + pulp.lpSum(E_storage_disch[RANGE,t])\
            - pulp.lpSum(E_storage_ch[RANGE,t])\
            + E_unserve[t]\
            - E_dump[t] == parasys["elec_load"][t] - parasys["solar_kW"][t]

    add_constraint("ElecBalance", index_hour, ElecBalance)

    def HeatBalance(index):
        # [t=1:H_t]
        t = index[0]
        return Q_HRUheating_out[t] + pulp.lpSum(boiler_x[RANGE,t]) + Heat_unserve[t] - Heat_dump[t] == parasys["heat_load"][t]
    add_constraint("HeatBalance",index_hour,HeatBalance)

    def CoolBalance(index):
        # [t=1:H_t]
        t = index[0]
        return pulp.lpSum(abs_x[RANGE,t]) + pulp.lpSum(chiller_x[RANGE,t]) + pulp.lpSum(Cool_storage_disch[RANGE,t]) - pulp.lpSum(Cool_storage_ch[RANGE,t]) + Cool_unserve[t] - Cool_dump[t] == parasys["cool_load"][t]
    add_constraint("CoolBalance",index_hour,CoolBalance)

    E_storage0 = np.zeros(N_E_storage)
    for i in range(N_E_storage):
        E_storage0[i] = 0.5*E_storage_para[i].Emax

    E_storageend = np.zeros(N_E_storage)
    E_storageend[0] = 628.87

    index_e_storage = (range(N_E_storage),)
    def E_storage_init(index):
        # [i=1:N_E_storage]
        i = index[0]
        return E_storage_state[i,1] == E_storage0[i] + E_storage_para[i].eta_ch * E_storage_ch[i,1]- 1/E_storage_para[i].eta_disch * E_storage_disch[i,1]
    add_constraint("E_storage_init", index_e_storage, E_storage_init)

    index_without_first_hour = (range(1,H_t),)
    def E_storage_state_constraint(index):
        # [i=1:N_E_storage,t=2:H_t]
        i, t = index
        return E_storage_state[i,t] == E_storage_state[i,t-1] + E_storage_para[i].eta_ch * E_storage_ch[i,t]- 1/E_storage_para[i].eta_disch * E_storage_disch[i,t]

    add_constraint("E_storage_state_constraint", index_e_storage + index_without_first_hour, E_storage_state_constraint)

    def E_storage_final(index):
        # [i=1:N_E_storage]
        i = index[0]
        return E_storage_state[i,H_t-1] >= E_storageend[i]
    add_constraint("E_storage_final", index_e_storage, E_storage_final)

    Cool_storage0 = np.zeros(N_Cool_storage)
    for i in range(N_Cool_storage):
        Cool_storage0[i] = 0.5 * Cool_storage_para[i].Emax

    Cool_storageend = np.zeros(N_Cool_storage)
    Cool_storageend[0] = 15.647

    index_cool_storage = (range(N_Cool_storage),)
    def Cool_storage_init(index):
        # [i=1:N_Cool_storage]
        i = index[0]
        return Cool_storage_state[i,1] == Cool_storage0[i] + Cool_storage_para[i].eta_ch * Cool_storage_ch[i,1]- 1/Cool_storage_para[i].eta_disch * Cool_storage_disch[i,1]
    add_constraint("Cool_storage_init", index_cool_storage, Cool_storage_init)

    def Cool_storage_state_constraint(index):
        # [i=1:N_Cool_storage,t=2:H_t]
        i, t = index
        return Cool_storage_state[i,t] == Cool_storage_state[i,t-1] + Cool_storage_para[i].eta_ch * Cool_storage_ch[i,t]- 1/Cool_storage_para[i].eta_disch * Cool_storage_disch[i,t]
    add_constraint("Cool_storage_state_constraint", index_cool_storage + index_without_first_hour, Cool_storage_state_constraint)

    def Cool_storage_final(index):
        # [i=1:N_Cool_storage]
        i = index[0]
        return Cool_storage_state[i,H_t-1] >= Cool_storageend[i]
    add_constraint("Cool_storage_final", index_cool_storage, Cool_storage_final)

    index_turbine = (turbine_names,)
    def turbineyConsume(index):
        # [i=1:N_turbine,t=1:H_t]
        i, t = index

        return turbine_y[i,t] == turbine_para[i].fundata["a"] * turbine_x_k[i,t,RANGE] + turbine_para[i].fundata["b"] * turbine_s_k[i,t,RANGE]
    add_constraint("turbineyConsume",index_turbine + index_hour,turbineyConsume)

    def turbinexGenerate(index):
        # [i=1:N_turbine,t=1:H_t]
        i, t = index
        return turbine_x[i,t] == pulp.lpSum(turbine_x_k[i,t,RANGE])
    add_constraint("turbinexGenerate", index_turbine + index_hour, turbinexGenerate)

    index_kk = (range(KK),)
    def turbinexlower(index):
        # [i=1:N_turbine,t=1:H_t,k=1:KK]
        i, t, k = index
        return turbine_para[i].fundata["min"][k] * turbine_s_k[i,t,k] <= turbine_x_k[i,t,k]
    add_constraint("turbinexlower", index_turbine + index_hour + index_kk, turbinexlower)

    def turbinexupper(index):
        # [i=1:N_turbine,t=1:H_t,k=1:KK]
        i, t, k = index
        return turbine_para[i].fundata["max"][k] * turbine_s_k[i,t,k] >= turbine_x_k[i,t,k]
    add_constraint("turbinexupper", index_turbine + index_hour + index_kk, turbinexupper)

    def turbinexstatus(index):
        # [i=1:N_turbine,t=1:H_t]
        i, t = index
        return turbine_s[i,t] == pulp.lpSum(turbine_s_k[i,t,RANGE])
    add_constraint("turbinexstatus", index_turbine + index_hour, turbinexstatus)

    def turbinestartstatus1(index):
        # [i=1:N_turbine,t=1]
        i, t = index[0], 0
        return turbine_start[i,t] >= turbine_s[i,1] - turbine_init[i].status
    add_constraint("turbinestartstatus1",index_turbine, turbinestartstatus1)

    def turbinestartstatus(index):
        # [i=1:N_turbine,t=2:H_t]
        i, t = index
        return turbine_start[i,t] >= turbine_s[i,t] - turbine_s[i,t-1]
    add_constraint("turbinestartstatus",index_turbine + index_without_first_hour, turbinestartstatus)

    def turbineramp1(index):
        # [i=1:N_turbine,t=1]
        i, t = index[0], 0
        return turbine_init[i].output + turbine_para[i].ramp_down  <= turbine_x[i,t] <= turbine_init[i].output + turbine_para[i].ramp_up
    add_constraint("turbineramp1", index_turbine, turbineramp1)

    def turbinerampup(index):
        # [i=1:N_turbine,t=2:H_t]
        i, t = index
        return turbine_x[i,t-1] + turbine_para[i].ramp_down  <= turbine_x[i,t]
    add_constraint("turbinerampup", index_turbine + index_without_first_hour, turbinerampup)

    def turbinerampdown(index):
        # [i=1:N_turbine,t=2:H_t]
        i, t = index
        return turbine_x[i,t] <= turbine_x[i,t-1] + turbine_para[i].ramp_up
    add_constraint("turbinerampdown", index_turbine + index_without_first_hour, turbinerampdown)

    def turbineslockon1(index):
        # [i=1:N_turbine,t=1]
        i, t = index[0], 0
        partial = []
        for tau in range(t-1):
            partial.apend(turbine_s[i,tau])
        return turbine_para[i].min_on * (turbine_init[i].status1[-1] - turbine_s[i,t]) <= pulp.lpSum(partial) + pulp.lpSum(turbine_init[i].status1[24+t-turbine_para[i].min_on:24])
    add_constraint("turbineslockon1", index_turbine, turbineslockon1)

    ################################################################################
    # These turbine constraints are extra weird. Their index variables refer to one
    # another so I'm not going to try forcing them into the add constraint utility

    for i in turbine_names:
        for t in range(1, turbine_para[i].min_on):
            name = "turbineslockon2_{}_{}".format(i, t)
            partial = []
            for tau in range(1, t-1):
                partial.append(turbine_s[i,tau])

            c = turbine_para[i].min_on * (turbine_s[i,t-1] - turbine_s[i,t]) <= pulp.lpSum(partial) + pulp.lpSum(turbine_init[i].status1[24+t-turbine_para[i].min_on:24])
            constraints.append((c, name))

    for i in turbine_names:
        for t in range(turbine_para[i].min_on+1, H_t):
            name = "turbineslockon_{}_{}".format(i, t)
            partial = []
            for tau in range(t-turbine_para[i].min_on, t-1):
                partial.append(turbine_s[i,tau])

            c = turbine_para[i].min_on * (turbine_s[i,t-1] - turbine_s[i,t]) <= pulp.lpSum(partial)
            constraints.append((c, name))

    ################################################################################

    index_boiler = (boiler_names,)
    def boileryConsume(index):
        # [i=1:N_boiler,t=1:H_t]
        i, t = index
        return boiler_y[i,t] ==boiler_para[i].fundata["a"] * boiler_x_k[i,t,RANGE] + boiler_para[i].fundata["b"] * boiler_s_k[i,t,RANGE]
    add_constraint("boileryConsume", index_boiler + index_hour, boileryConsume)

    def boilerxGenerate(index):
        # [i=1:N_boiler,t=1:H_t]
        i, t = index
        return boiler_x[i,t] == pulp.lpSum(boiler_x_k[i,t,RANGE])
    add_constraint("boilerxGenerate", index_boiler + index_hour, boilerxGenerate)

    def boilerxlower(index):
        # [i=1:N_boiler,t=1:H_t,k=1:KK]
        i, t, k = index
        return boiler_para[i].fundata["min"][k] * boiler_s_k[i,t,k] <= boiler_x_k[i,t,k]
    add_constraint("boilerxlower", index_boiler + index_hour + index_kk, boilerxlower)

    def boilerxupper(index):
        # [i=1:N_boiler,t=1:H_t,k=1:KK]
        i, t, k = index
        return boiler_para[i].fundata["max"][k] * boiler_s_k[i,t,k] >= boiler_x_k[i,t,k]
    add_constraint("boilerxupper",index_boiler + index_hour + index_kk, boilerxupper)

    def boilerxstatus(index):
        # [i=1:N_boiler,t=1:H_t]
        i, t = index
        return boiler_s[i,t] == pulp.lpSum(boiler_s_k[i,t,RANGE])
    add_constraint("boilerxstatus", index_boiler + index_hour, boilerxstatus)

    def boilerstartstatus1(index):
        # [i=1:N_boiler,t=1]
        i, t = index[0], 0
        return boiler_start[i,t] >= boiler_s[i,1] - boiler_init[i].status
    add_constraint("boilerstartstatus1", index_boiler, boilerstartstatus1)

    def boilerstartstatus(index):
        # [i=1:N_boiler,t=2:H_t]
        i, t = index
        return boiler_start[i,t] >= boiler_s[i,t] - boiler_s[i,t-1]
    add_constraint("boilerstartstatus", index_boiler + index_without_first_hour, boilerstartstatus)

    def boilerramp1(index):
        # [i=1:N_boiler,t=1]
        i, t = index[0], 0
        return boiler_init[i].output + boiler_para[i].ramp_down  <= boiler_x[i,t] <= boiler_init[i].output + boiler_para[i].ramp_up
    add_constraint("boilerramp1", index_boiler, boilerramp1)

    def boilerrampup(index):
        # [i=1:N_boiler,t=2:H_t]
        i, t = index
        return boiler_x[i,t-1] + boiler_para[i].ramp_down  <= boiler_x[i,t]
    add_constraint("boilerrampup", index_boiler + index_without_first_hour, boilerrampup)

    def boilerrampdown(index):
        # [i=1:N_boiler,t=2:H_t]
        i, t = index
        return boiler_x[i,t] <= boiler_x[i,t-1] + boiler_para[i].ramp_up
    add_constraint("boilerrampdown",index_boiler + index_without_first_hour, boilerrampdown)

    index_chiller = (chiller_names,)
    def chilleryConsume(index):
        # [i=1:N_chiller,t=1:H_t]
        i, t = index
        return chiller_y[i,t] ==chiller_para[i].fundata["a"] * chiller_x_k[i,t,RANGE] + chiller_para[i].fundata["b"] *chiller_s_k[i,t,RANGE]
    add_constraint("chilleryConsume", index_chiller + index_hour, chilleryConsume)

    def chillerxGenerate(index):
        # [i=1:N_chiller,t=1:H_t]
        i, t = index
        return chiller_x[i,t] == pulp.lpSum(chiller_x_k[i,t,RANGE])
    add_constraint("chillerxGenerate", index_chiller + index_hour, chillerxGenerate)

    def chillerxlower(index):
        # [i=1:N_chiller,t=1:H_t,k=1:KK]
        i, t, k = index
        return chiller_para[i].fundata["min"][k] * chiller_s_k[i,t,k] <= chiller_x_k[i,t,k]
    add_constraint("chillerxlower", index_chiller + index_hour + index_kk, chillerxlower)

    def chillerxupper(index):
        # [i=1:N_chiller,t=1:H_t,k=1:KK]
        i, t, k = index
        return chiller_para[i].fundata["max"][k] * chiller_s_k[i,t,k] >= chiller_x_k[i,t,k]
    add_constraint("chillerxupper", index_chiller + index_hour + index_kk, chillerxupper)

    def chillerxstatus(index):
        # [i=1:N_chiller,t=1:H_t]
        i, t = index
        return chiller_s[i,t] == pulp.lpSum(chiller_s_k[i,t,RANGE])
    add_constraint("chillerxstatus", index_chiller + index_hour,chillerxstatus)

    def chillerstartstatus1(index):
        # [i=1:N_chiller,t=1]
        i, t = index[0], 0
        return chiller_start[i,t] >= chiller_s[i,1] - chiller_init[i].status
    add_constraint("chillerstartstatus1", index_chiller, chillerstartstatus1)

    def chillerstartstatus(index):
        # [i=1:N_chiller,t=2:H_t]
        i, t = index
        return chiller_start[i,t] >= chiller_s[i,t] - chiller_s[i,t-1]
    add_constraint("chillerstartstatus", index_chiller + index_without_first_hour, chillerstartstatus)

    def chillerramp1(index):
        # [i=1:N_chiller,t=1]
        i, t = index[0], 0
        return chiller_init[i].output + chiller_para[i].ramp_down  <= chiller_x[i,t] <= chiller_init[i].output + chiller_para[i].ramp_up
    add_constraint("chillerramp1", index_chiller, chillerramp1)

    def chillerrampup(index):
        # [i=1:N_chiller,t=2:H_t]
        i, t = index
        return chiller_x[i,t-1] + chiller_para[i].ramp_down  <= chiller_x[i,t]
    add_constraint("chillerrampup", index_chiller + index_without_first_hour, chillerrampup)

    def chillerrampdown(index):
        # [i=1:N_chiller,t=2:H_t]
        i, t = index
        return chiller_x[i,t] <= chiller_x[i,t-1] + chiller_para[i].ramp_up
    add_constraint("chillerrampdown", index_chiller + index_without_first_hour, chillerrampdown)

    index_abs = (abs_names,)
    def absyConsume(index):
        # [i=1:N_abs,t=1:H_t]
        i, t = index
        return abs_y[i,t] ==abs_para[i].fundata["a"] * abs_x_k[i,t,RANGE] + abs_para[i].fundata["b"] *abs_s_k[i,t,RANGE]
    add_constraint("absyConsume",index_abs + index_hour, absyConsume)

    def absxGenerate(index):
        # [i=1:N_abs,t=1:H_t]
        i, t = index
        return abs_x[i,t] == pulp.lpSum(abs_x_k[i,t,RANGE])
    add_constraint("absxGenerate", index_abs + index_hour, absxGenerate)

    def absxlower(index):
        # [i=1:N_abs,t=1:H_t,k=1:KK]
        i, t, k = index
        return abs_para[i].fundata["min"][k] * abs_s_k[i,t,k] <= abs_x_k[i,t,k]
    add_constraint("absxlower", index_abs + index_hour + index_kk, absxlower)

    def absxupper(index):
        # [i=1:N_abs,t=1:H_t,k=1:KK]
        i, t, k = index
        return abs_para[i].fundata["max"][k] * abs_s_k[i,t,k] >= abs_x_k[i,t,k]
    add_constraint("absxupper", index_abs + index_hour + index_kk, absxupper)

    def absxstatus(index):
        # [i=1:N_abs,t=1:H_t]
        i, t = index
        return abs_s[i,t] == pulp.lpSum(abs_s_k[i,t,RANGE])
    add_constraint("absxstatus", index_abs + index_hour, absxstatus)

    def absstartstatus1(index):
        # [i=1:N_abs,t=1]
        i, t = index[0], 0
        return abs_start[i,t] >= abs_s[i,1] - abs_init[i].status
    add_constraint("absstartstatus1", index_abs, absstartstatus1)

    def absstartstatus(index):
        # [i=1:N_abs,t=2:H_t]
        i, t = index
        return abs_start[i,t] >= abs_s[i,t] - abs_s[i,t-1]
    add_constraint("absstartstatus", index_abs + index_without_first_hour, absstartstatus)

    def absramp1(index):
        # [i=1:N_abs,t=1]
        i, t = index[0], 0
        return abs_init[i].output + abs_para[i].ramp_down  <= abs_x[i,t] <= abs_init[i].output + abs_para[i].ramp_up
    add_constraint("absramp1", index_abs, absramp1)

    def absrampup(index):
        # [i=1:N_abs,t=2:H_t]
        i, t = index
        return abs_x[i,t-1] + abs_para[i].ramp_down  <= abs_x[i,t]
    add_constraint("absrampup", index_abs + index_without_first_hour, absrampup)

    def absrampdown(index):
        # [i=1:N_abs,t=2:H_t]
        i, t = index
        return abs_x[i,t] <= abs_x[i,t-1] + abs_para[i].ramp_up
    add_constraint("absrampdown",index_abs + index_without_first_hour, absrampdown)

    abs_0 = abs_names[0]
    turbine_0 = turbine_names[0]
    turbine_1 = turbine_names[1]
    def wastedheat(index):
        # [t=1:H_t]
        t = index[0]
        return Q_HRUheating_in[t] + abs_y[abs_0, t] == turbine_y[turbine_0,t] - turbine_x[turbine_0,t]/293.1 + turbine_y[turbine_1,t] - turbine_x[turbine_1,t]/293.1
    add_constraint("wastedheat", index_hour, wastedheat)

    def HRUlimit(index):
        # [t=1:H_t]
        t = index[0]
        return Q_HRUheating_out[t] <= a_hru * Q_HRUheating_in[t]
    add_constraint("HRUlimit", index_hour, HRUlimit)



    objective_components = []

    for var, _lambda in zip(E_gridelecfromgrid[RANGE], parasys["electricity_cost"]):
        objective_components.append(var * _lambda)

    for var, _lambda in zip(E_gridelectogrid[(RANGE,)], parasys["electricity_cost"]):
        objective_components.append(var * _lambda)

    for i in range(2):
        for var, _lambda in zip(turbine_y[i, RANGE], parasys["natural_gas_cost"]):
            objective_components.append(var * _lambda)


    # for var, _lambda in zip(turbine_y[(2, RANGE)], parasys["lambda_diesel"]):
    #     objective_components.append(var * _lambda)

    for i in boiler_names:
        for var, _lambda in zip(boiler_y[i, RANGE],parasys["natural_gas_cost"]):
            objective_components.append(var * _lambda)

    for i in turbine_names:
        for var in turbine_start[i, RANGE]:
            objective_components.append(var * turbine_para[i].startcost)

    for i in boiler_names:
        for var in boiler_start[i, RANGE]:
            objective_components.append(var * boiler_para[i].startcost)

    for i in chiller_names:
        for var in chiller_start[i, RANGE]:
            objective_components.append(var * chiller_para[i].startcost)

    for i in abs_names:
        for var in abs_start[i, RANGE]:
            objective_components.append(var * abs_para[i].startcost)

    for group in (E_unserve, E_dump, Heat_unserve, Heat_dump, Cool_unserve, Cool_dump):
        for var in group[RANGE]:
            objective_components.append(var * bigM)


    prob = pulp.LpProblem("Building Optimization", pulp.LpMinimize)
    prob += pulp.lpSum(objective_components), "Objective Function"

    for c in constraints:
        prob += c

    return prob
