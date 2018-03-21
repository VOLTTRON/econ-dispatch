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
import os
import numpy as np
import pandas as pd

from econ_dispatch.component_models import ComponentBase
from econ_dispatch.utils import least_squares_regression

FARADAY_CONSTANT = 96485 # C/mol

class Coefs(object):
    pass

Coef = Coefs()
Coef.NominalCurrent = np.array([0.1921, 0.1582, 0.0261])
Coef.NominalASR = 0.5
Coef.ReStartDegradation = float(1e-3)
Coef.LinearDegradation = float(4.5e-6)
Coef.ThresholdDegradation = float(8e3)
Coef.Utilization = np.array([-.25, -.2, .65])
Coef.StackHeatLoss = 0.1
Coef.AncillaryPower = np.array([0.5, 4.0, 0.25])
Coef.Area = 5000.0 #5000cm^2 and 100 cells producing 100kW works out to 0.2 W/cm^2 and at a voltage of 0.6 this is 0.333 amp/cm^2
Coef.StackDeltaT = 100.0
Coef.ExhaustTemperature = np.array([0.0, 0.0, 0.0])
Coef.gain = 1.4

EXPECTED_PARAMETERS = set(["xmin",
                           "xmax",
                           "mat",
                           "cap"])

class Component(ComponentBase):
    def __init__(self, capacity=500.0,
                 fuel_type="CH4",
                 nominal_power=402.0,
                 nominal_ocv=0.8,
                 **kwargs):
        super(Component, self).__init__(**kwargs)
        self.capacity = capacity
        self.fuel_type = fuel_type
        self.nominal_power = nominal_power
        self.nominal_ocv = nominal_ocv

        #self.power = 300
        #self.T_amb = 20.0
        #self.gen_start = 5.0
        #self.gen_hours = 7000.0
        
    def get_output_metadata(self):
        return [u"electricity", u"waste_heat"]

    def get_input_metadata(self):
        return [u"natural_gas"]

    # def validate_parameters(self):
    #     k = set(self.parameters.keys())
    #     return EXPECTED_PARAMETERS <= k

    def train(self, training_data):

        # Valid = Power > 1% of cap and AmbTemperature > 5.0
        # TODO: replace valid with this
        Valid = training_data['Valid']

        Power = training_data['Power']
        Power = Power[Valid]

        AmbTemperature = training_data['AmbTemperature']
        AmbTemperature = AmbTemperature[Valid]

        Start = training_data['Start']
        Start = Start[Valid]

        Hours = training_data['Hours']
        Hours = Hours[Valid]

        FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency = self.FuelCell_Operate(Coef, Power, AmbTemperature,
                                                                                         Start, Hours)

        sort_indexes = np.argsort(Power)
        Xdata = Power[sort_indexes]
        Ydata = FuelFlow[sort_indexes] * 171.11  # fuel: kg/s -> mmBtu/hr

        xmin = min(Xdata)
        xmax = max(Xdata)

        n1 = np.nonzero(Xdata <= xmin + (xmax - xmin) * 0.3)[-1][-1]
        n2 = np.nonzero(Xdata <= xmin + (xmax - xmin) * 0.6)[-1][-1]

        mat = least_squares_regression(inputs=Xdata[n1:n2 + 1], output=Ydata[n1:n2 + 1])

        self.parameters = {
            "mat": mat.tolist(),
            "xmax": xmax,
            "xmin": xmin,
            "cap": self.capacity
        }

    def get_mapped_commands(self, component_loads):
        return {"set_point":
                component_loads["Q_prime_mover_{}_hour00".format(self.name)]*293.1}


    # def get_optimization_parameters(self):
    #     if not self.opt_params_dirty:
    #         return self.cached_parameters.copy()
    #
    #     Valid = self.training_data['Valid'].values
    #
    #     Power = self.training_data['Power'].values
    #     Power = Power[Valid]
    #
    #     AmbTemperature = self.training_data['AmbTemperature'].values
    #     AmbTemperature = AmbTemperature[Valid]
    #
    #     Start = self.training_data['Start'].values
    #     Start = Start[Valid]
    #
    #     Hours = self.training_data['Hours'].values
    #     Hours = Hours[Valid]
    #
    #     FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency = self.FuelCell_Operate(Coef, Power, AmbTemperature, Start, Hours)
    #
    #     sort_indexes = np.argsort(Power)
    #     Xdata = Power[sort_indexes]
    #     Ydata = FuelFlow[sort_indexes] * 171.11 # fuel: kg/s -> mmBtu/hr
    #
    #     xmin = min(Xdata)
    #     xmax = max(Xdata)
    #
    #     n1 = np.nonzero(Xdata <= xmin + (xmax - xmin) * 0.3)[-1][-1]
    #     n2 = np.nonzero(Xdata <= xmin + (xmax - xmin) * 0.6)[-1][-1]
    #
    #     mat = least_squares_regression(inputs=Xdata[n1:n2+1], output=Ydata[n1:n2+1])
    #
    #     self.cached_parameters = {
    #         "mat": mat.tolist(),
    #         "xmax": xmax,
    #         "xmin": xmin,
    #         "cap_prime_mover": self.capacity
    #     }
    #
    #     self.opt_params_dirty = False
    #     return self.cached_parameters.copy()



    def FuelCell_Operate(self, Coef, Power, Tin, Starts, NetHours):
        if self.fuel_type.lower() == "ch4":
            n = 8 # number of electrons per molecule (assuming conversion to H2)
            LHV = 50144 # Lower heating value of CH4 in kJ/g
            m_fuel = 16 # molar mass
        elif self.fuel_type.lower() == "h2":
            n = 2 # number of electrons per molecule (assuming conversion to H2)
            LHV = 120210 # Lower heating value of H2 in kJ/kmol
            m_fuel = 2# molar mass
        else:
            raise ValueError("Unknown fuel type {}".format(self.fuel_type))

        cells = self.nominal_power

        nPower = Power / self.nominal_power
        ASR = Coef.NominalASR + Coef.ReStartDegradation * Starts + Coef.LinearDegradation * max(0,np.amax(NetHours - Coef.ThresholdDegradation)) #ASR  in Ohm*cm^2 
        Utilization = Coef.Utilization[0] * (1 - nPower)**2 + Coef.Utilization[1] * (1 - nPower) + Coef.Utilization[2] #decrease in utilization at part load
        Current = Coef.Area * (Coef.NominalCurrent[0] * nPower**2 + Coef.NominalCurrent[1] * nPower + Coef.NominalCurrent[2]) #first guess of current
        HeatLoss = Power * Coef.StackHeatLoss
        AncillaryPower = 0.1 * Power
        for _ in xrange(4):
            Voltage = cells * (self.nominal_ocv - Current * ASR / Coef.Area)
            Current = Coef.gain * (Power + AncillaryPower) * 1000 / Voltage - (Coef.gain - 1) * Current
            FuelFlow = m_fuel * cells * Current / (n * 1000 * FARADAY_CONSTANT * Utilization)
            ExhaustFlow = (cells * Current * (1.2532 - Voltage / cells) / 1000 - HeatLoss) / (1.144 * Coef.StackDeltaT) #flow rate in kg/s with a specific heat of 1.144kJ/kg*K
            AncillaryPower = Coef.AncillaryPower[0] * FuelFlow**2 + Coef.AncillaryPower[1] * FuelFlow + Coef.AncillaryPower[0] * ExhaustFlow**2 + Coef.AncillaryPower[1] * ExhaustFlow + Coef.AncillaryPower[2] * (Tin - 18) * ExhaustFlow

        ExhaustTemperature = ((cells * Current * (1.2532 - Voltage / cells) / 1000 - HeatLoss) + (1 - Utilization) * FuelFlow * LHV) / (1.144 * ExhaustFlow) + Tin + (Coef.ExhaustTemperature[0] * nPower**2 + Coef.ExhaustTemperature[1]*nPower + Coef.ExhaustTemperature[2])
        NetEfficiency = Power / (FuelFlow * LHV)

        return FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency
