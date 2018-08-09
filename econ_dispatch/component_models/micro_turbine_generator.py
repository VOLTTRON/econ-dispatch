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


import json
from math import ceil, isnan
import os
import numpy as np

from econ_dispatch.component_models import ComponentBase
from econ_dispatch import utils
import logging

_log = logging.getLogger(__name__)

class Coefs(object):
    pass

# Coef = Coefs()
# Coef.NominalPower = 60.0
# Coef.Fuel_LHV = 50144.0
# Coef.TempDerateThreshold = 15.556 #from product specification sheet
# Coef.TempDerate = 0.12 # (#/C) from product specification sheet
# Coef.Maintenance = 3.0 # in percent/year
# Coef.HeatLoss = 2.0 / 3.0
# Coef.Eff = np.array([-0.2065, 0.3793, 0.1043])
# Coef.FlowOut = np.array([-65.85, 164.5])

EXPECTED_PARAMETERS = set(["fundata",
                            "ramp_up",
                            "ramp_down",
                            "start_cost",
                            "min_on",
                            "output"
])

class Component(ComponentBase):
    def __init__(self,
                 fuel_lhv=50144.0,
                 temp_derate_threshold=None,
                 temp_derate=None,
                 ramp_up=None,
                 ramp_down=None,
                 start_cost=None,
                 min_on=0,
                 **kwargs):
        super(Component, self).__init__(**kwargs)

        # training_data = os.path.join(os.path.dirname(__file__), 'CapstoneTurndownData.json')
        # with open(training_data_file, 'r') as f:
        #     capstone_turndown_data = json.load(f)
        #
        # self.Pdemand = np.array(capstone_turndown_data['Pdemand'])
        # self.Temperature = np.array(capstone_turndown_data['Temperature']) - 273
        # FuelFlow = np.array(capstone_turndown_data['FuelFlow'])
        # AirFlow = np.array(capstone_turndown_data['AirFlow'])
        self.coef = Coefs()
        self.coef.NominalPower = self.capacity # Capacity must be included
        self.coef.Fuel_LHV = float(fuel_lhv)
        self.coef.TempDerateThreshold = float(temp_derate_threshold) #from product specification sheet
        self.coef.TempDerate = float(temp_derate) # (#/C) from product specification sheet
        #self.coef.Maintenance = 3.0 # in percent/year # Too far into the weeds - Nick
        self.coef.HeatLoss = 2.0 / 3.0 # Calculated later
        self.coef.Eff = np.array([-0.2065, 0.3793, 0.1043]) # Calculated later
        self.coef.FlowOut = np.array([-65.85, 164.5]) # Calculated later

        self.min_on = min_on

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.start_cost = start_cost
        self.output = 0

        self.command_history = [0] * 24

        # self.parameters["cap"] = self.coef.NominalPower


    def get_output_metadata(self):
        return [u"electricity", u"waste_heat"]

    def get_input_metadata(self):
        return [u"natural_gas"]

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        try:
            set_point = component_loads["turbine_x_{}_0".format(self.name)]
        except KeyError:
            set_point = component_loads["Q_prime_mover_{}_hour00".format(self.name)]

        self.output = set_point
        self.parameters["output"] = self.output

        self.command_history = self.command_history[1:] + [int(set_point>0)]
        self.parameters["command_history"] = self.command_history[:]

        return {"set_point": set_point}

    training_inputs_name_map = {
        "outputs": "power",
        "inputs": "fuel_flow"
    }

    def train(self, training_data):
        Power = training_data['power']
        # Temperature = training_data['temperature'] - 273
        FuelFlow = training_data['fuel_flow']
        # AirFlow = training_data['air_flow']
        # Time = np.array([])
        # self.coef = self.GasTurbine_Calibrate(self.coef, Time, Power, Temperature, FuelFlow, AirFlow, np.array([]))
        # AirFlow, FuelFlow, Tout, Efficiency = self.GasTurbine_Operate(Power, Temperature, 0, self.coef)
        #
        # sort_indexes = np.argsort(Power)
        # Xdata = Power[sort_indexes]
        # Ydata = FuelFlow[sort_indexes] * 171.11  # fuel: kg/s -> mmBtu/hr
        #
        # xmin = min(Xdata)
        # xmax = max(Xdata)
        #
        # n1 = np.nonzero(Xdata <= xmin + (xmax - xmin) * 0.3)[-1][-1]
        # n2 = np.nonzero(Xdata <= xmin + (xmax - xmin) * 0.6)[-1][-1]
        #
        # # mat = least_squares_regression(inputs=Xdata[n1:n2 + 1], output=Ydata[n1:n2 + 1])
        # #
        # # self.parameters = {
        # #     "mat": mat.tolist(),
        # #     "xmax": xmax,
        # #     "xmin": xmin,
        # #     "cap": self.coef.NominalPower
        # # }
        #
        # a, b, xmin, xmax = utils.piecewise_linear(Xdata[n1:n2 + 1], Ydata[n1:n2 + 1], self.coef.NominalPower)

        Xdata = Power
        Ydata = FuelFlow * 171.11  # fuel: kg/s -> mmBtu/hr

        timestamps = training_data.get("timestamps", None)

        try:
            inputs, outputs = utils.clean_training_data(Ydata, Xdata, self.capacity)
        except ValueError as err:
            _log.debug("Training data does not meet standards: {}".format(err))
            inputs, outputs = utils.get_default_curve("micro_turbine_generator", self.capacity, 0.35,
                                                        timestamps=timestamps)

        # a, b, xmin, xmax = utils.piecewise_linear(inputs, outputs, self.capacity,
        #                                           curve_func=lambda x, p0, p1, p2, p3: p0*x/(p1+p2*x+p3*x**2))
        a, b, xmin, xmax = utils.piecewise_linear(inputs, outputs, self.capacity)

        self.parameters = {
            "fundata": {
                "a": a,
                "b": b,
                "min": xmin,
                "max": xmax
            },
            "ramp_up": self.ramp_up,
            "ramp_down": self.ramp_down,
            "start_cost": self.start_cost,
            "min_on": self.min_on,
            "output": self.output,
            "command_history": self.command_history[:]
        }

        _log.debug("Fuel cell {} parameters: {}".format(self.name, self.parameters))

    # def GasTurbine_Operate(self, Power, Tin, NetHours, Coef):
    #     #Pdemand in kW
    #     #Tin in C
    #     #Net Hours in hours since last maintenance event
    #     Tderate = Coef.TempDerate / 100.0 * (Tin - Coef.TempDerateThreshold)
    #     Tderate[Tderate < 0] = 0
    #     # MaintenanceDerate = NetHours / 8760.0 * Coef.Maintenance / 100#efficiency scales linearly with hours since last maintenance.
    #     MaintenanceDerate = 0
    #     Pnorm = Power / Coef.NominalPower
    #
    #     Efficiency = (Coef.Eff[0] * Pnorm**2 + Coef.Eff[1] * Pnorm + Coef.Eff[2]) - Tderate - MaintenanceDerate
    #     FuelFlow = Power / (Efficiency * Coef.Fuel_LHV)
    #     AirFlow = FuelFlow * (Coef.FlowOut[0] * Pnorm + Coef.FlowOut[1]) #air mass flow rate in kg/s
    #     Tout = Tin + (FuelFlow * Coef.Fuel_LHV - (1 + Coef.HeatLoss) * Power) / (1.1 * AirFlow) #flow rate in kg/s with a specific heat of 1.1kJ/kg*K
    #
    #     return AirFlow, FuelFlow, Tout, Efficiency
    #
    # def GasTurbine_Calibrate(self, Coef, Time, Power, Temperature, FuelFlow, AirFlow, ExhaustTemperature):
    #     #This function determines the best fit coefficients for modeling a gas turbine
    #     ## User can provide the following Coeficients, or they can be calculated here:
    #     # 'NominalPower': The nominal power of the turbine
    #     # 'Fuel_LHV': Lower heating value of the fuel in kJ/kg
    #     # 'TempDerateThreshold': The temperature (C) at which the turbine efficiency starts to decline
    #     # 'TempDerate': The rate at which efficiency declines (#/C) after the threshold temperature
    #     # 'Maintenance': The rate of performance decline between maintenance cycles (#/yr)
    #     ## Other inputs are:
    #     # Time (cumulative hours of operation)
    #     # Power (kW)
    #     # Temperature (C) ambient
    #     # FuelFlow (kg/s)
    #     # AirFlow (kg/s) (Optional)
    #     # Exhaust Temperature (C)  (Optional)
    #
    #     try:
    #         Coef.NominalPower
    #     except AttributeError:
    #         P2 = sorted(Power)
    #         Coef.NominalPower = P2[int(ceil(0.98 * len(Power)))]
    #
    #     try:
    #         Coef.Fuel_LHV
    #     except AttributeError:
    #         Coef.Fuel_LHV = 50144.0 # Assume natural gas with Lower heating value of CH4 in kJ/kg
    #
    #     Eff = Power / (FuelFlow * Coef.Fuel_LHV)
    #     Eff = np.nan_to_num(Eff) # zero out any bad FuelFlow data
    #
    #     # Much of this block may need reevaluation.
    #     # It doesn't work with with the available inputs
    #     try:
    #         Coef.TempDerate
    #     except AttributeError:
    #         Tsort = sorted(Temperature)
    #
    #         # The assignments may not be the min and max of the sorted array
    #         Tmin = Tsort[int(ceil(0.1 * len(Tsort)))]
    #         Tmax = Tsort[int(ceil(0.9 * len(Tsort)))]
    #         C = np.zeros(10)
    #         # 'valid' can be all False values causing np.polyfit to fail
    #         for i in range(10):
    #             # minus one might need to be changed for python zero indexing?
    #             tl = Tmin + (i - 1) / 10 * (Tmax - Tmin)
    #             th = Tmin + i / 10 * (Tmax - Tmin)
    #             valid = (Eff > 0) * (Eff < 0.5) * (Temperature > tl) * (Temperature < th)
    #             fit = np.polyfit(Temperature[valid], Eff[valid]*100, 1)
    #             C[i] = -fit[1]
    #
    #         #negligable dependence on temperature
    #         C[C < 0.025] = 0
    #
    #         try:
    #             I = [c for c in C if c > 0.1][0]
    #             Coef.TempDerateThreshold = Tmin + I / 10 * (Tmax - Tmin)
    #             Coef.TempDerate = np.mean(C[C>0])
    #         except IndexError:
    #             Coef.TempDerateThreshold = 20
    #             Coef.TempDerate = 0
    #
    #     # This will not work if time is not passed
    #     # try:
    #     #     Coef.Maintenance
    #     # except AttributeError:
    #     #     valid = (Eff > 0) * (Eff < 0.5)
    #     #     fit = np.polyfit(Time[valid]/8760, Eff[valid]*100, 1) #time in yrs, eff in %
    #     #     Coef.Maintenance = -fit(1)
    #     #
    #     Tderate = Coef.TempDerate * (Temperature - Coef.TempDerateThreshold)
    #     Tderate[Tderate < 0] = 0
    #     #
    #     # if Time.size != 0:
    #     #     MaintenanceDerate = Time / 8760.0 * Coef.Maintenance
    #     # else:
    #     #     MaintenanceDerate = 0
    #
    #     # remove temperature dependence from data prior to fit
    #     Eff = Eff + Tderate / 100 #+ MaintenanceDerate / 100
    #
    #     # efficiency of gas turbine before generator conversion
    #     Coef.Eff = np.polyfit(Power/Coef.NominalPower, Eff, 2)
    #
    #     if AirFlow.size > 0:
    #         Coef.FlowOut = np.polyfit(Power / Coef.NominalPower, AirFlow / FuelFlow, 1)
    #     elif ExhaustTemperature.size > 0: #assume a heat loss and calculate air mass flow
    #         AirFlow = (FuelFlow * Coef.Fuel_LHV - 1.667 * Power) / (1.1 * (ExhaustTemperature - Temperature))
    #         Coef.FlowOut = np.polyfit(Power / Coef.NominalPower, AirFlow/FuelFlow, 1)
    #     else:
    #         # assume a flow rate
    #         Coef.FlowOut = np.array([-65.85, 164.5])
    #
    #     if ExhaustTemperature.size > 0:
    #         # flow rate in kg/s with a specific heat of 1.1kJ/kg*K
    #         Coef.HeatLoss =  np.mean(((FuelFlow * Coef.Fuel_LHV - Power) - (ExhaustTemperature - Temperature) * 1.1 * AirFlow) / Power)
    #     else:
    #         Coef.HeatLoss = 2.0 / 3.0
    #
    #     return Coef
