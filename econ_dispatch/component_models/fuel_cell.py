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
import numpy as np

from econ_dispatch.component_models import ComponentBase

FARADAY_CONSTANT = 96485 # C/mol

class Coefs(object):
    pass

Coef = Coefs()
Coef.NominalCurrent = np.array([0.1921, 0.1582, 0.0261])
Coef.NominalASR = 0.5
Coef.ReStartDegradation = float(1e-3)
Coef.LinearDegradation = float(4.5e-6)
Coef.ThresholdDegradation = float(9e3)#hours before which there is no linear degradation
Coef.Utilization = np.array([-.25, -.2, .65])
Coef.StackHeatLoss = 0.1
Coef.AncillaryPower = np.array([0.5, 4.0, 0.25])
Coef.Area = 5000.0 #5000cm^2 and 100 cells producing 100kW works out to 0.2 W/cm^2 and at a voltage of 0.6 this is 0.333 amp/cm^2
Coef.StackDeltaT = 100.0
Coef.ExhaustTemperature = np.array([0.0, 0.0, 0.0])
Coef.gain = 1.4

# NominalV = np.array([0.775, 0.8, 0.814])

class Component(ComponentBase):
    def __init__(self):
        super(Component, self).__init__()
        self.fuel_type= 'CH4'
        self.nominal_power = 203.0
        self.nominal_ocv = 0.775

        self.power = 300.0
        self.T_amb = 20.0
        self.gen_start = 5.0
        self.gen_hours = 7000.0
        
    def get_output_metadata(self):
        return ""

    def get_input_metadata(self):
        return ""

    def get_optimization_parameters(self):
        FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency = self.FuelCell_Operate(Coef)
        return {}

    def update_parameters(self):
        pass

    def FuelCell_Operate(self, Coef):
        if self.fuel_type == "CH4":
            n = 8 # number of electrons per molecule (assuming conversion to H2)
            LHV = 50144 # Lower heating value of CH4 in kJ/g
            m_fuel = 16 # molar mass
        elif self.fuel_type == "H2":
            n = 2 # number of electrons per molecule (assuming conversion to H2)
            LHV = 120210 # Lower heating value of H2 in kJ/kmol
            m_fuel = 2# molar mass
        else:
            raise ValueError("Unknown fuel type {}".format(self.fuel_type))
    
        cells = self.nominal_power
        power = self.power
        T_amb = self.T_amb
        gen_start = self.gen_start
        gen_hours = self.gen_hours
        
        n_power = power / self.nominal_power
        ASR = Coef.NominalASR + Coef.ReStartDegradation * gen_start + Coef.LinearDegradation * max(0, (gen_hours - Coef.ThresholdDegradation)) #ASR  in Ohm*cm^2
        Utilization = Coef.Utilization[0] * (1 - n_power)**2 + Coef.Utilization[1] * (1 - n_power) + Coef.Utilization[2] #decrease in utilization at part load
        Current = Coef.Area * (Coef.NominalCurrent[0] * n_power**2 + Coef.NominalCurrent[1] * n_power + Coef.NominalCurrent[2]) #first guess of current
        HeatLoss = power * Coef.StackHeatLoss
        Ancillarypower = 0.1 * power
    
        for j in range(4):
            Voltage = cells * (Coef.NominalOCV - Current * ASR / Coef.Area)
            Current = Coef.gain * (power + Ancillarypower) * 1000 / Voltage - (Coef.gain - 1) * Current
            FuelFlow = m_fuel * cells * Current / (n * 1000 * FARADAY_CONSTANT * Utilization)
            ExhaustFlow = (cells * Current * (1.2532 - Voltage / cells) / 1000 - HeatLoss) / (1.144 * Coef.StackDeltaT) #flow rate in kg/s with a specific heat of 1.144kJ/kg*K
            Ancillarypower = Coef.Ancillarypower[0] * FuelFlow**2  + Coef.Ancillarypower[1] * FuelFlow + Coef.Ancillarypower[0] * ExhaustFlow**2 + Coef.Ancillarypower[1]*ExhaustFlow + Coef.Ancillarypower[2] * (T_amb - 18) * ExhaustFlow
    
        ExhaustTemperature = ((cells * Current *(1.2532 - Voltage / cells) / 1000 - HeatLoss) + (1 - Utilization) * FuelFlow * LHV) / (1.144 * ExhaustFlow) + T_amb + (Coef.ExhaustTemperature[0]*n_power**2 + Coef.ExhaustTemperature[1] * n_power + Coef.ExhaustTemperature[2])
        NetEfficiency = power / (FuelFlow * LHV)
        
        return FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency
