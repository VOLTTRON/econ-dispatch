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
from CoolProp.HumidAirProp import HAPropsSI

from econ_dispatch.component_models import ComponentBase

# specific heat of water in J/g-C
CP_WATER = 4.186

DEFAULT_CFM_OA = 1000
DEFAULT_FAN_STATUS = 1
DEFAULT_T_OA = 32
DEFAULT_RH_OA = 0.6
DEFAULT_T_HW = 46.1

class Component(ComponentBase):
    def __init__(self, **kwargs):
        super(Component, self).__init__(**kwargs)

        GetTraining = True

        # An optional input is the hot water outlet temperature from the regeneration coil.
        # This data is used only in a training function to provide a better estimate of the detla_T in the coil, compared to default assumption of 5 degrees C.
        # This training also requires an input of the hot water coil valve command, which is used as a filter of the data for building the regression
        T_HW_out_available = True
        
        ### DEFAULT ASSUMPTIONS
        # Electric power in Watts consumed to run desiccant system, typically including desiccant wheel motor and heat recovery wheel motor
        self.wheel_power = 300
        
        # Assumed HW temperature drop across Regeneration Coil, if HW outlet temperature is not measured
        self.deltaT_RegenCoil = 5
        
        ### REQUIRED INPUTS
        # CFM of outdoor air.  This is expected to be a constant value when the fan is on (minimum outdoor air flow rate)
        self.cfm_OA = DEFAULT_CFM_OA
        
        # Fan status for air-hanling unit with desiccant system (1= ON, 0 = OFF)
        self.fan_status = DEFAULT_FAN_STATUS
        
        # outdoor air temperature, from building sensor, in degrees C
        self.T_OA = DEFAULT_T_OA
        
        # outdoor air relative humidity, from building sensor in kg/m3
        self.RH_OA = DEFAULT_RH_OA
        
        # Hot water loop temperature in degrees C
        self.T_HW = DEFAULT_T_HW
        
        
        if GetTraining:
            TrainingData = pd.read_csv('SampleDesiccant.csv', header=0)
            w_oa_min = HAPropsSI('W', 'T', (11.67+273.15), 'P', 101325, 'R', 1) #This is an estimate of the minimum humidy ratio down to which dehumidifcation is useful in relieving cooling coil latent cooling loads.  The assumed humidity conditions are saturation (100% RH) at 53 degrees F, which is a typical cooling coil setpoint for dehumidification
            h_oa_min = HAPropsSI('H', 'T', (11.67+273.15), 'P', 101325, 'R', 1) #corresponding minimum enthalpy
            print h_oa_min
            self.Coefs = getDesiccantCoeffs(TrainingData, h_oa_min) # regression is a funciton of MODEL variables, rather than AHU system variables, since the AHU system is not modeled.  specific regression variables are outdoor air enthalpy, outdoor air enthalpy squared, hot water inlet temperature, and outdoor air CFM (which may be a constant value when the fan is on)
        
            # If hot water outlet temperature from regeneration coil is available, use historical data on hot water coil inlet and outlet temperatures along with valve command to build regression.
            if T_HW_out_available:
                self.Coefs2 = getTrainingMFR(TrainingData)
    
    def get_output_metadata(self):
        return ""

    def get_input_metadata(self):
        return ""

    def get_optimization_parameters(self):
        Q_Cool_Out, Q_HW_In, Q_Elec_In, MFR_HW_In, T_HW_Out = self.predict()
        return {}

    def update_parameters(self,
                          cfm_OA=DEFAULT_CFM_OA,
                          fan_status=DEFAULT_FAN_STATUS,
                          T_OA=DEFAULT_T_OA,
                          RH_OA=DEFAULT_RH_OA,
                          T_HW=DEFAULT_T_HW):

        self.cfm_OA = cfm_OA
        self.fan_status = fan_status
        self.T_OA = T_OA
        self.RH_OA = RH_OA
        self.T_HW = T_HW

    def getTrainingMFR(TrainingData):
        #This function is used by the Desiccant Wheel program to estimate the delta T across the regeneration coil for the purpose of calculating the hot water mass flow rate in the regeneration coil
        # This funciton is used only when historical data for the regneration coil outlet tempreature is available.
    
        # Outdoor air temperature sensor; convert to degrees C
        T_OA = TrainingData['T_OA'].values
    
        # Outdoor air relative humudity (fraction)
        RH_OA = TrainingData['RH_OA'].values
    
        # regeneration coil valve command (fraction)
        Vlv = TrainingData['Regen_Valve'].values
    
        # Hot water inlet temperature to regeneration coil [C]
        T_hw = TrainingData['T_HW'].values
    
        # Hot water outlet temperature from regeneration coil [C]
        T_hw_out = TrainingData['T_HW_out'].values

        # Fan status (1=ON, 0=OFF)
        Fan = TrainingData['FanStatus'].values

        Rows = len(T_hw_out)
    
        h_OA_fan = []
    
        # hot water temperature drop across regneration coil [C]
        delta_T = []
    
        for i in range(0, Rows):
            if Fan[i] > 0 and Vlv[i] > 0:
                h_OAx = HAPropsSI('H', 'T', (T_OA[i]+273.15), 'P', 101325, 'R', RH_OA[i]/100); #calculate outdoor air enthalpy
                h_OA_fan.append(h_OAx)
                delta_T.append(T_hw[i] - T_hw_out[i])
    
        #single variabel regression based on outdoor air enthalpy
        y = delta_T
        ones = np.ones(len(h_OA_fan))
        X = np.column_stack((h_OA_fan, ones))
        Coefficients, resid, rank, s = np.linalg.lstsq(X, y)

        return Coefficients
    
    def getDesiccantCoeffs(TrainingData, h_oa_min):
    
        #This function is used by th Desiccant Wheel program and uses historical trend data on outdoor and conditioned air temperuatures and relative humidities, outdoor air flow rates, fan status and outdoor air temperature
        # A regression dequation is built to predict the drop in enthalpy of the supply air acrsoss the desiccant coil as a function of outdoor air enthalpy, outdoor enthalpy squared, hot water temperature and CFM of outdoor air
    
        #timestamp
        Time = TrainingData['Date/Time'].values
    
        # Outdoor air temperature sensor; convert to degrees C
        T_OA = TrainingData['T_OA'].values
    
        # Outdoor air relative humudity (fraction)
        RH_OA = TrainingData['RH_OA'].values
    
        # Conditioned temperature of supply air at the outlet of the heat recovery coil
        T_CA = TrainingData['T_CA'].values
        
        # Volumetric flow rate of outdor air in cubic feet per minute (cfm)
        CFM_OA = TrainingData['CFM_OA'].values
    
        # Conditioned relative humidity of supply air at the outlet of the heat recovery coil
        RH_CA = TrainingData['RH_CA'].values
    
        # hot water inlet temperature to regeneration coil
        T_hw = TrainingData['T_HW'].values
    
        # Fan status (1=ON, 0=OFF)
        Fan = TrainingData['FanStatus'].values
    
        Rows = len(Time)
    
        h_OA = []
        h_CA = []
        delta_h = []
        h_OA_fan = []
        h_OA2_fan = []
        T_HW_fan = []
        CFM_OA_fan = []
    
    
        for i in range(0, Rows):
            if Fan[i] > 0:
                h_OAx = HAPropsSI('H','T', (T_OA[i]+273.15), 'P', 101325, 'R', RH_OA[i]/100) #calculate outdoor air enthalpy
                if  h_OAx > h_oa_min:
                    h_OA.append(h_OAx)
                    h_CAx = HAPropsSI('H', 'T', (T_CA[i]+273.15), 'P', 101325, 'R', RH_CA[i]/100)
                    h_CA.append(h_CAx)
                    delta_h.append(h_CAx-h_OAx)
                    h_OA_fan.append(h_OAx)
                    h_OA2_fan.append(pow(h_OAx, 2))
                    T_HW_fan.append(T_hw[i])
                    CFM_OA_fan.append(CFM_OA[i])
    
        # regression variables are hot water inlet temperature, outdoor air enthalpy, outdoor air enthalpy squared and CFM of outdoor air
        y = delta_h
        ones = np.ones(len(T_HW_fan))
        X = np.column_stack((CFM_OA_fan, h_OA2_fan, h_OA_fan, T_HW_fan, ones))
        Coefficients, resid, rank, s = np.linalg.lstsq(X, y)

        return Coefficients


    def predict():
        # outdoor air enthalpy in current timestep
        h_oa_curr = HAPropsSI('H', 'T', (self.T_OA + 273.15), 'P', 101325, 'R', self.RH_OA)
        
        # outdoor air humidity ratio in current timestep
        w_oa_curr = HAPropsSI('W', 'T', (self.T_OA + 273.15), 'P', 101325, 'R', self.RH_OA)
        
        # If hot water outlet temperature from regeneration coil is available, get estimate of delta T as a funciton of outdoor air enthalpy.
        if T_HW_out_available:
            self.deltaT_RegenCoil = self.Coefs2[0] * h_oa_curr + self.Coefs2[1]
        
        # Thermal COP is the reduction in cooling load divided by the heat input to the desiccant wheel. The regression equation here is based on Figure 6 in http://www.nrel.gov/docs/fy05osti/36974.pdf
        COP_thermal = 0.2482 + 4.171529 * w_oa_curr - 0.00019 * self.T_OA + 0.000115 * pow(self.T_OA, 2)
        
        # Calculate the change in enthalpy from the outdoor air to the conditioned air stream, prior to mixing box (or cooling coil if it is a 100% OA unit)
        deltaEnthalpy = self.Coefs[4] + self.T_HW * self.Coefs[3] + h_oa_curr * self.Coefs[2] + pow(h_oa_curr,2) * self.Coefs[1] + self.cfm_OA * self.Coefs[0]
        
        # No useful enthalpy reduction if outdoor air humidity ratio is already below saturated conditions at the cooling coil outlet temperature used for dehumidification
        if w_oa_curr < w_oa_min or self.fan_status == 0: 
            deltaEnthalpy = 0
        # Useful enthalpy reduction should be limited by enthalpy of the supply air at saturation for the cooling coil in dehumidification mode
        elif deltaEnthalpy < (h_oa_min - h_oa_curr):
            deltaEnthalpy = h_oa_min - h_oa_curr
        
        
        # density of outdoor air in kg/m3 as a function of outdoor air temperature in degrees C
        rho_OA = 0.0000175 * pow(self.T_OA, 2) - (0.00487 * self.T_OA) + 1.293
        
        # calculate mass flow rate of outdoor air, given volumetric flow rate and air density
        MFR_OA= self.cfm_OA * rho_OA / pow(3.28, 3) / 60
        
        # output: reduction in cooling load (W)
        Q_Cool_Out = -deltaEnthalpy * MFR_OA
        
        # output: hot water loop heat input (W)
        Q_HW_In = Q_Cool_Out / COP_thermal
        
        #output: mass flow rate of water in heating coil [kg/s]
        MFR_HW_In = Q_HW_In / CP_WATER / self.deltaT_RegenCoil / 1000
        
        #output: hot water temperature outlet from regeneration coil [C]
        T_HW_Out = self.T_HW - self.deltaT_RegenCoil * self.fan_status
        
        #output: Electricity Input [W]
        Q_Elec_In = self.wheel_power
        if Q_Cool_Out == 0:
            Q_Elec_In = 0

        return Q_Cool_Out, Q_HW_In, Q_Elec_In, MFR_HW_In, T_HW_Out
        
