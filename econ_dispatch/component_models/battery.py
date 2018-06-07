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

import os
import pandas as pd
import numpy as np

from econ_dispatch.component_models import ComponentBase
from econ_dispatch.utils import least_squares_regression

import logging

_log = logging.getLogger(__name__)

DEFAULT_CAPACITY = 24000
DEFAULT_INPUT_POWER_REQUEST = -3000
DEFAULT_TIMESTEP = 500
DEFAULT_PREVSOC = 0.3996
DEFAULT_INPUTCURRENT = -12.5
DEFAULT_MINIMUMSOC = 0.4
DEFAULT_INVEFF_CHARGE = 0.94
DEFAULT_IR_CHARGE = 0.7
DEFAULT_INVEFF_DISCHARGE = 0.94
DEFAULT_IR_DISCHARGE = 0.7
DEFAULT_IDLE_A = -0.00012
DEFAULT_IDLE_B = -0.00024

EXPECTED_PARAMETERS = set(["soc",
                           "charge_eff",
                           "discharge_eff",
                           "min_power",
                           "max_power",
                           "min_soc",
                           "max_soc",
                           "cap"])

class Component(ComponentBase):
    def __init__(self,
                 min_power=None,
                 max_power=None,
                 min_soc=0.3,
                 max_soc=0.8,
                 **kwargs):
        super(Component, self).__init__(**kwargs)
        self.min_power = float(min_power)
        self.max_power = float(max_power)
        self.min_soc = float(min_soc)
        self.max_soc = float(max_soc)
        self.current_soc = None

        self.parameters["min_power"] = self.min_power
        self.parameters["max_power"] = self.max_power
        self.parameters["min_soc"] = self.min_soc
        self.parameters["max_soc"] = self.max_soc
        self.parameters["cap"] = self.capacity

    def validate_parameters(self):
        k = set(self.parameters.keys())
        return EXPECTED_PARAMETERS <= k and self.parameters["soc"] is not None

    def get_mapped_commands(self, component_loads):
        try:
            charge_load = component_loads["E_storage_ch_{}_0".format(self.name)]
            discharge_load = component_loads["E_storage_disch_{}_0".format(self.name)]
        except KeyError:
            _log.warning("battery load missing from optimizer output")
            return {}
        return {"charge_load": charge_load, "discharge_load": discharge_load}

    def process_input(self, timestamp, name, value):
        """Override this to process input data from the platform.
        Components will typically want the current state of the device they
        represent as input.
        name - Name of the input from the configuration file.
        value - value of the input from the message bus.
        """
        if name == "soc":
            self.current_soc = value
            self.parameters["soc"] = value

    def train(self, training_data):
        charge_eff = self.calculate_charge_eff(training_data, True)
        discharge_eff = self.calculate_charge_eff(training_data, False)

        self.parameters["soc"] = self.current_soc
        self.parameters["charge_eff"] = charge_eff
        self.parameters["discharge_eff"] = discharge_eff
        self.parameters["min_power"] = self.min_power
        self.parameters["max_power"] = self.max_power
        self.parameters["min_soc"] = self.min_soc
        self.parameters["max_soc"] = self.max_soc
        self.parameters["cap"] = self.capacity

    def calculate_charge_eff(self, charge_training_data, charging):
        timestamp = charge_training_data['timestamp']
        PowerIn = charge_training_data['power']
        SOC = charge_training_data['soc']

        # Skip chunks where we do not charge.
        if charging:
            a = PowerIn[1:  ] > 0
            b = PowerIn[ :-1] > 0
        else:
            a = PowerIn[1:  ] < 0
            b = PowerIn[ :-1] < 0
        valid = np.insert(a&b, 0, False).nonzero()[0]
        valid_prev = valid - 1

        prev_soc = SOC[valid_prev]
        current_soc = SOC[valid]

        delta_soc = current_soc - prev_soc

        delta_kWh = delta_soc * self.capacity

        prev_time = timestamp[valid_prev]
        current_time = timestamp[valid]
        delta_time = current_time - prev_time

        # Convert delta_time to fractional hours
        delta_time = delta_time.astype("timedelta64[s]").astype("float64")/3600.0

        current_power = PowerIn[valid]
        if charging:
            eff = (delta_kWh) / (current_power * delta_time)
        else:
            eff = (current_power * delta_time) / (delta_kWh)

        # Remove garbage values.
        valid_eff = eff[eff<1.0]
        eff_avg = abs(valid_eff.mean())

        _log.debug("calculate_charge_eff charging {} result {}, dropped {} values before calculation".format(charging, eff_avg, len(eff)-len(valid_eff)))

        return eff_avg


    # def get_optimization_parameters(self):
    #     NewSOC, InputPower = self.getUpdatedSOC()
    #     return {"SOC": NewSOC, "InputPower": InputPower}


    # def update_parameters(self, timestamp, inputs):
    #     self.capacity = inputs.get("capacity", DEFAULT_CAPACITY)
    #     self.input_power_request = inputs.get("input_power_request", DEFAULT_INPUT_POWER_REQUEST)
    #     self.timestep = inputs.get("timestep", DEFAULT_TIMESTEP)
    #     self.PrevSOC = inputs.get("PrevSOC", DEFAULT_PREVSOC)
    #     self.InputCurrent = inputs.get("InputCurrent", DEFAULT_INPUTCURRENT)
    #     self.minimumSOC = inputs.get("minimumSOC", DEFAULT_MINIMUMSOC)
    #     self.InvEFF_Charge = inputs.get("InvEFF_Charge", DEFAULT_INVEFF_CHARGE)
    #     self.IR_Charge = inputs.get("IR_Charge", DEFAULT_IR_CHARGE)
    #     self.InvEFF_DisCharge = inputs.get("InvEFF_DisCharge", DEFAULT_INVEFF_DISCHARGE)
    #     self.IR_DisCharge = inputs.get("IR_DisCharge", DEFAULT_IR_DISCHARGE)
    #     self.Idle_A = inputs.get("Idle_A", DEFAULT_IDLE_A)
    #     self.Idle_B = inputs.get("Idle_B", DEFAULT_IDLE_B)


    # def getUpdatedSOC(self):
    #
    #     # change in state of charge during this self.timestep. Initialization of variable
    #     deltaSOC = 0
    #
    #     #for batttery charging
    #     #P2 is power recieved into battery storage
    #     if self.input_power_request > 0:
    #         P2 = self.input_power_request * self.InvEFF_Charge - self.InputCurrent * self.InputCurrent * self.self.IR_Charge
    #         deltaSOC = P2*self.timestep/3600/self.capacity
    #
    #     elif self.input_power_request < 0:
    #         P2 = self.input_power_request/self.InvEFF_DisCharge-self.InputCurrent*self.InputCurrent*self.IR_DisCharge
    #         deltaSOC = P2*self.timestep/3600/self.capacity
    #
    #
    #     deltaSOC_self = (self.Idle_A + self.Idle_B * self.PrevSOC) * self.timestep / 3600
    #     SOC = self.PrevSOC + deltaSOC + deltaSOC_self
    #     InputPower = self.input_power_request
    #
    #     if SOC < self.minimumSOC:
    #         if self.PrevSOC < self.minimumSOC and self.input_power_request < 0:
    #             InputPower = 0
    #             SOC = self.PrevSOC + deltaSOC_self
    #         if self.PrevSOC > self.minimumSOC and self.input_power_request < 0:
    #             InputPower = self.input_power_request * (self.PrevSOC - self.minimumSOC) / (self.PrevSOC - SOC)
    #             self.InputCurrent = self.InputCurrent * InputPower / self.input_power_request
    #             P2 = InputPower / self.InvEFF_DisCharge - self.InputCurrent * self.InputCurrent * self.IR_DisCharge
    #             deltaSOC= P2 * self.timestep / 3600 / self.capacity
    #             SOC = self.PrevSOC + deltaSOC + deltaSOC_self
    #     if SOC > 1:
    #             InputPower = self.input_power_request * (1 - self.PrevSOC) / (SOC - self.PrevSOC)
    #             self.InputCurrent = self.InputCurrent * InputPower / self.input_power_request
    #             P2 = InputPower * self.InvEFF_Charge - self.InputCurrent * self.InputCurrent * self.self.IR_Charge
    #             deltaSOC= P2 * self.timestep / 3600 / self.capacity
    #             SOC = self.PrevSOC + deltaSOC + deltaSOC_self
    #     if SOC < 0:
    #             SOC = 0
    #
    #
    #     return SOC, InputPower
    #
    # def GetChargingParameters(self, charging_training_file):
    #     # data_file = os.path.join(os.path.dirname(__file__), 'BatteryCharging.csv')
    #     TrainingData = pd.read_csv(charging_training_file, header=0)
    #     Time = TrainingData['Time'].values
    #     Current = TrainingData['I'].values
    #     PowerIn = TrainingData['Po'].values
    #     SOC = TrainingData['SOC'].values
    #     x = len(Time)
    #     PrevTime = []
    #     CurrTime = []
    #     CurrPower = []
    #     PrevPower = []
    #     CurrSOC = []
    #     self.PrevSOC = []
    #     CurrentI = []
    #     ChargeEff = []
    #     Slope_RI = []
    #
    #     for i in range(0, x + 1):
    #         if i>1:
    #             PrevTime.append(Time[i-2])
    #             CurrTime.append(Time[i-1])
    #             CurrPower.append(PowerIn[i-1])
    #             CurrSOC.append(SOC[i-1])
    #             self.PrevSOC.append(SOC[i-2])
    #             CurrentI.append(Current[i-1])
    #
    #     for i in range(0,x-1):
    #         ChargeEff.append((CurrSOC[i] - self.PrevSOC[i]) * self.capacity / ((CurrTime[i] - PrevTime[i]) * 24) / CurrPower[i])
    #         Slope_RI.append(0 - CurrentI[i] * CurrentI[i] / CurrPower[i])
    #
    #     x = Slope_RI
    #     y = ChargeEff
    #
    #     intercept, slope = least_squares_regression(inputs=x, output=y)
    #     return intercept, slope
    #
    # def GetDisChargingParameters(self, discharging_training_file):
    #     # data_file = os.path.join(os.path.dirname(__file__), 'BatteryDisCharging.csv')
    #     TrainingData = pd.read_csv(discharging_training_file, header=0)
    #     Time = TrainingData['Time'].values
    #     Current = TrainingData['I'].values
    #     PowerIn = TrainingData['Po'].values
    #     SOC = TrainingData['SOC'].values
    #     Rows = len(Time)
    #     PrevTime = []
    #     CurrTime = []
    #     CurrPower= []
    #     PrevPower = []
    #     CurrSOC = []
    #     self.PrevSOC = []
    #     CurrentI = []
    #     Y = []
    #     X = []
    #
    #     for i in range(0, Rows + 1):
    #         if i > 1:
    #             PrevTime.append(Time[i-2])
    #             CurrTime.append(Time[i-1])
    #             CurrPower.append(PowerIn[i-1])
    #             CurrSOC.append(SOC[i-1])
    #             self.PrevSOC.append(SOC[i-2])
    #             CurrentI.append(Current[i-1])
    #
    #     for i in range(0,Rows-1):
    #         Y.append((CurrSOC[i] - self.PrevSOC[i]) * self.capacity / ((CurrTime[i] - PrevTime[i]) * 24) / (CurrentI[i] * CurrentI[i]))
    #         X.append(CurrPower[i] / (CurrentI[i] * CurrentI[i]))
    #
    #     intercept, slope = least_squares_regression(inputs=X, output=Y)
    #
    #     IR_discharge = (0 - intercept)
    #     InvEFFDischarge = 1 / slope
    #     return IR_discharge, InvEFFDischarge
    #
    # def GetIdleParameters(self, idle_training_file):
    #     # data_file = os.path.join(os.path.dirname(__file__), 'BatteryIdle.csv')
    #     TrainingData = pd.read_csv(idle_training_file, header=0)
    #     Time = TrainingData['Time'].values
    #     SOC = TrainingData['SOC'].values
    #     Rows = len(Time)
    #     PrevTime = []
    #     CurrTime = []
    #     CurrSOC = []
    #     self.PrevSOC = []
    #     Y = []
    #     X = []
    #
    #     for i in range(0, Rows + 1):
    #         if i > 1:
    #             PrevTime.append(Time[i-2])
    #             CurrTime.append(Time[i-1])
    #             CurrSOC.append(SOC[i-1])
    #             self.PrevSOC.append(SOC[i-2])
    #
    #     for i in range(0, Rows - 1):
    #         Y.append((CurrSOC[i] - self.PrevSOC[i]) / ((CurrTime[i] - PrevTime[i]) * 24))
    #         X.append(CurrSOC[i])
    #
    #     intercept, slope = least_squares_regression(inputs=X, output=Y)
    #
    #     return slope, intercept
    
