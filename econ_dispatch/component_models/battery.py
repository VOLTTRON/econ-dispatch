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

import pandas as pd
import numpy as np

from econ_dispatch.component_models import ComponentBase

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


class Component(ComponentBase):
    def __init__(self, **kwargs):
        super(Component, self).__init__(**kwargs)
        #Battery Storage Capacity in W-h
        self.capacity = DEFAULT_CAPACITY

        #Input: Requested Power to battery in W.  Positive for charging, negative for discharging. 0 for idle
        # Request may be truncated to an actual Input power if minimum SOC or maximum SOC (1.0) limits are reached
        self.input_power_request = DEFAULT_INPUT_POWER_REQUEST

        #timestep in seconds
        self.timestep = DEFAULT_TIMESTEP

        #State of Charge in previous self.timestep; normalized 0 to 1
        self.PrevSOC = DEFAULT_PREVSOC

        # Amperes
        self.InputCurrent = DEFAULT_INPUTCURRENT

        # limit depth of discharge to prolong battery life
        self.minimumSOC = DEFAULT_MINIMUMSOC

        # internal inverter efficiency from AC to DC during battery charging.
        self.InvEFF_Charge = DEFAULT_INVEFF_CHARGE

        # battery internal resistance during charging, in Ohms.
        self.IR_Charge = DEFAULT_IR_CHARGE

        # internal inverter efficiency from AC to DC during battery charging.
        self.InvEFF_DisCharge = DEFAULT_INVEFF_DISCHARGE

        # battery internal resistance during charging, in Ohms.
        self.IR_DisCharge = DEFAULT_IR_DISCHARGE

        # constant rate of SOC loss due to battery self-discharge (change in SOC per hour)
        self.Idle_A = DEFAULT_IDLE_A

        # rate of SOC loss multiplied by current SOC; due to battery self-discharge
        self.Idle_B = DEFAULT_IDLE_B

        RunTrainingCharge = True
        RunTrainingDisCharge = True
        RunTrainingIdle = False

        if RunTrainingCharge:
            C = self.GetChargingParameters(self.capacity)
            self.InvEFF_Charge = C[0]
            self.self.IR_Charge = C[1]

        if RunTrainingDisCharge:
            C = self.GetDisChargingParameters(self.capacity)
            self.IR_DisCharge = C[0]
            self.InvEFF_DisCharge = C[1]

        if RunTrainingIdle:
            C = self.GetIdleParameters(self.capacity)
            self.Idle_A = C[1]
            self.Idle_B = C[0]

    def get_output_metadata(self):
        return ""

    def get_input_metadata(self):
        return ""

    def get_optimization_parameters(self):
        NewSOC, InputPower = self.getUpdatedSOC()
        return {}

    def update_parameters(self,
                          capacity=DEFAULT_CAPACITY,
                          input_power_request=DEFAULT_INPUT_POWER_REQUEST,
                          timestep=DEFAULT_TIMESTEP,
                          PrevSOC=DEFAULT_PREVSOC,
                          InputCurrent=DEFAULT_INPUTCURRENT,
                          minimumSOC=DEFAULT_MINIMUMSOC,
                          InvEFF_Charge=DEFAULT_INVEFF_CHARGE,
                          IR_Charge=DEFAULT_IR_CHARGE,
                          InvEFF_DisCharge=DEFAULT_INVEFF_DISCHARGE,
                          IR_DisCharge=DEFAULT_IR_DISCHARGE,
                          Idle_A=DEFAULT_IDLE_A,
                          Idle_B=DEFAULT_IDLE_B):

        self.capacity = capacity
        self.input_power_request = input_power_request
        self.timestep = timestep
        self.PrevSOC = PrevSOC
        self.InputCurrent = InputCurrent
        self.minimumSOC = minimumSOC
        self.InvEFF_Charge = InvEFF_Charge
        self.IR_Charge = IR_Charge
        self.InvEFF_DisCharge = InvEFF_DisCharge
        self.IR_DisCharge = IR_DisCharge
        self.Idle_A = Idle_A
        self.Idle_B = Idle_B


    def getUpdatedSOC(self):

        # change in state of charge during this self.timestep. Initialization of variable    
        deltaSOC = 0
    
        #for batttery charging
        #P2 is power recieved into battery storage
        if self.input_power_request > 0:
            P2 = self.input_power_request * self.InvEFF_Charge - self.InputCurrent * self.InputCurrent * self.self.IR_Charge
            deltaSOC = P2*self.timestep/3600/self.capacity 
            
        elif self.input_power_request < 0:
            P2 = self.input_power_request/self.InvEFF_DisCharge-self.InputCurrent*self.InputCurrent*self.IR_DisCharge
            deltaSOC = P2*self.timestep/3600/self.capacity
    
    
        deltaSOC_self = (self.Idle_A + self.Idle_B * self.PrevSOC) * self.timestep / 3600
        SOC = self.PrevSOC + deltaSOC + deltaSOC_self
        InputPower = self.input_power_request
    
        if SOC < self.minimumSOC:
            if self.PrevSOC < self.minimumSOC and self.input_power_request < 0:
                InputPower = 0
                SOC = self.PrevSOC + deltaSOC_self
            if self.PrevSOC > self.minimumSOC and self.input_power_request < 0:
                InputPower = self.input_power_request * (self.PrevSOC - self.minimumSOC) / (self.PrevSOC - SOC)
                self.InputCurrent = self.InputCurrent * InputPower / self.input_power_request
                P2 = InputPower / self.InvEFF_DisCharge - self.InputCurrent * self.InputCurrent * self.IR_DisCharge
                deltaSOC= P2 * self.timestep / 3600 / self.capacity
                SOC = self.PrevSOC + deltaSOC + deltaSOC_self
        if SOC > 1:
                InputPower = self.input_power_request * (1 - self.PrevSOC) / (SOC - self.PrevSOC)
                self.InputCurrent = self.InputCurrent * InputPower / self.input_power_request
                P2 = InputPower * self.InvEFF_Charge - self.InputCurrent * self.InputCurrent * self.self.IR_Charge
                deltaSOC= P2 * self.timestep / 3600 / self.capacity
                SOC = self.PrevSOC + deltaSOC + deltaSOC_self
        if SOC < 0:
                SOC = 0
            
        
        return SOC, InputPower
    
    def GetChargingParameters(self):
        TrainingData = pd.read_csv('BatteryCharging.csv', header=0)
        Time = TrainingData['Time'].values
        Current = TrainingData['I'].values
        PowerIn = TrainingData['Po'].values
        SOC = TrainingData['SOC'].values
        x = len(Time)
        PrevTime = []
        CurrTime = []
        CurrPower = []
        PrevPower = []
        CurrSOC = []
        self.PrevSOC = []
        CurrentI = []
        ChargeEff = []
        Slope_RI = []
    
        for i in range(0, x + 1):
            if i>1:
                PrevTime.append(Time[i-2])
                CurrTime.append(Time[i-1])
                CurrPower.append(PowerIn[i-1])
                CurrSOC.append(SOC[i-1])
                self.PrevSOC.append(SOC[i-2])
                CurrentI.append(Current[i-1])
    
        for i in range(0,x-1):
            ChargeEff.append((CurrSOC[i] - self.PrevSOC[i]) * self.capacity / ((CurrTime[i] - PrevTime[i]) * 24) / CurrPower[i])
            Slope_RI.append(0 - CurrentI[i] * CurrentI[i] / CurrPower[i])
    
        x = Slope_RI
        y = ChargeEff

        xs = np.column_stack((np.ones(len(x)), x))
        (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, y)

        return intercept, slope
    
    def GetDisChargingParameters(self):
        TrainingData = pd.read_csv('BatteryDisCharging.csv', header=0)
        Time = TrainingData['Time'].values
        Current = TrainingData['I'].values
        PowerIn = TrainingData['Po'].values
        SOC = TrainingData['SOC'].values
        Rows = len(Time)
        PrevTime = []
        CurrTime = []
        CurrPower= []
        PrevPower = []
        CurrSOC = []
        self.PrevSOC = []
        CurrentI = []
        Y = []
        X = []
    
        for i in range(0, Rows + 1):
            if i > 1:
                PrevTime.append(Time[i-2])
                CurrTime.append(Time[i-1])
                CurrPower.append(PowerIn[i-1])
                CurrSOC.append(SOC[i-1])
                self.PrevSOC.append(SOC[i-2])
                CurrentI.append(Current[i-1])
    
        for i in range(0,Rows-1):
            Y.append((CurrSOC[i] - self.PrevSOC[i]) * self.capacity / ((CurrTime[i] - PrevTime[i]) * 24) / (CurrentI[i] * CurrentI[i]))
            X.append(CurrPower[i] / (CurrentI[i] * CurrentI[i]))
    
        xs = np.column_stack((np.ones(len(X)), X))
        (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, Y)

        IR_discharge = (0 - intercept)
        InvEFFDischarge = 1 / slope
        return IR_discharge, InvEFFDischarge
    
    def GetIdleParameters(self):
        TrainingData = pd.read_csv('BatteryIdle.csv', header=0)
        Time = TrainingData['Time'].values
        SOC = TrainingData['SOC'].values
        Rows = len(Time)
        PrevTime = []
        CurrTime = []
        CurrSOC = []
        self.PrevSOC = []
        Y = []
        X = []
    
        for i in range(0, Rows + 1):
            if i > 1:
                PrevTime.append(Time[i-2])
                CurrTime.append(Time[i-1])
                CurrSOC.append(SOC[i-1])
                self.PrevSOC.append(SOC[i-2])
    
        for i in range(0, Rows - 1):
            Y.append((CurrSOC[i] - self.PrevSOC[i]) / ((CurrTime[i] - PrevTime[i]) * 24))
            X.append(CurrSOC[i])

        xs = np.column_stack((np.ones(len(X)), X))
        (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, Y)
    
        return slope, intercept
    
