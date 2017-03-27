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

import math

import pandas as pd
import numpy as np

from econ_dispatch.component_models import ComponentBase

DEFAULT_MFR_EX = 2
DEFAULT_MFR_WATER = 15
DEFAULT_T_EX_I = 311
DEFAULT_T_WATER_I = 60
DEFAULT_PRIME_MOVER_SPEED = 0.75


class Component(ComponentBase):
    def __init__(self, **kwargs):
        super(Component, self).__init__(**kwargs)
        # kg/s
        self.MFR_ex = DEFAULT_MFR_EX

        # kg/s
        self.MFR_water = DEFAULT_MFR_WATER

        # sensor input: exhaust fluid temperature inlet to heat recovery in degrees C
        self.T_ex_i = DEFAULT_T_EX_I

        # sensor input: building hot water loop fluid temperature inlet to heat recovery in degrees C
        self.T_water_i = DEFAULT_T_WATER_I

        # sensor input: current speed of CHP prime mover.  Used only for regression.  Input units are not important and can be a percent speed or a fuel input parameter
        self.prime_mover_current_speed = DEFAULT_PRIME_MOVER_SPEED

        self.prime_mover_fluid = 'FlueGas'
        self.hot_water_loop_fluid = 'Water'

        # m2; default.  Only used if Method == 1
        self.heat_exchanger_area = 8

        #Only used if Method == 2
        self.default_effectiveness = 0.9

        self.C = [0, 0]

        # Method 1 calls for input of the heat transfer area - which is sometimes found on spec sheets for HXs - then calculates the effectiveness based on NTU method
        # Method 2 is a simplified approach that calls for the user to specify a default heat exchanger effectiveness
        # Method 3 builds a regression equation to calculate the outlet water temperature based on training data
        self.eff_calculation_method = 3

        #Training data may only need to be used periodically to find regression coefficients,  which can thereafter be re-used
        NeedRegressionCoefficients = True
        if NeedRegressionCoefficients and self.eff_calculation_method == 3:
            self.C = GetRegressionHeatRecovery()

    def get_output_metadata(self):
        return ""

    def get_input_metadata(self):
        return ""

    def get_optimization_parameters(self):
        water_out_temp = HeatRecoveryWaterOut()
        heat_recovered = GetHeatRecovered(water_out_temp)
        return {}

    def update_parameters(self,
                          MFR_ex=DEFAULT_MFR_EX,
                          MFR_water=DEFAULT_MFR_WATER,
                          T_ex_i=DEFAULT_T_EX_I,
                          T_water_i=DEFAULT_T_WATER_I,
                          prime_mover_current_speed=DEFAULT_PRIME_MOVER_SPEED):
        self.MFR_ex = MFR_ex
        self.MFR_water = MFR_water
        self.T_ex_i = T_ex_i
        self.T_water_i = T_water_i
        self.prime_mover_current_speed = prime_mover_current_speed

    def GetRegressionHeatRecovery():
        TrainingData = pd.read_csv('MicroturbineData.csv',  header=0)
        Twi = TrainingData['T_wi'].values
        Two = TrainingData['T_wo'].values
        Texi = TrainingData['self.T_ex_i'].values
        Speed = TrainingData['Speed'].values
        MFRw = TrainingData['self.MFR_water'].values
        Proportional = Speed / MFRw * (Texi - Twi)
        Output = Two - Twi

        x = Proportional
        y = Output

        bias = np.ones(len(x))
        xs = np.column_stack((bias, x))
        (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, y)

        return intercept, slope

    def HeatRecoveryWaterOut():

        if self.prime_mover_fluid == 'FlueGas':
            cp_ex = 1002 #j/kg-K
        elif self.prime_mover_fluid == 'Water':
            cp_ex = 4186 #j/kg-K
        elif self.prime_mover_fluid == 'GlycolWater30':
            cp_ex = 3913 #j/kg-K
        elif self.prime_mover_fluid == 'GlycolWater50':
            cp_ex = 3558 #j/kg-K
        else:
            cp_ex = 4186 #j/kg-K

        if self.hot_water_loop_fluid == 'Water':
            cp_water = 4186 #j/kg-K
        elif self.hot_water_loop_fluid == 'GlycolWater30':
            cp_water = 3913 #j/kg-K
        elif self.hot_water_loop_fluid == 'GlycolWater50':
            cp_water = 3558 #j/kg-K
        else:
            cp_water = 4186 #j/kg-K

        C_ex = self.MFR_ex * cp_ex
        C_water = self.MFR_water * cp_water
        C_min = min(C_ex, C_water)
        C_max = max(C_ex, C_water)
        C_r = C_min / C_max

        if self.eff_calculation_method == 1:
            U = 650 #w/m2-K
            UA = U * self.heat_exchanger_area #W-K
            NTU = UA / C_min
            Eff = (1 - math.exp(-1 * NTU * (1 - C_r))) / (1 - C_r * math.exp(-1 * NTU * (1 - C_r))) #relation for concentric tube in counterflow
            T_water_o = self.T_water_i + Eff * C_min / C_water * (self.T_ex_i - self.T_water_i)
        elif self.eff_calculation_method == 2:
            Eff = self.default_effectiveness
            T_water_o = self.T_water_i + Eff * C_min / C_water * (self.T_ex_i - self.T_water_i)
        elif self.eff_calculation_method == 3:
            T_water_o = self.C[0] + self.T_water_i + self.C[1] * self.prime_mover_current_speed / self.MFR_water * (self.T_ex_i - self.T_water_i)

        return T_water_o

    def GetHeatRecovered(water_out_temp):
        if self.hot_water_loop_fluid == 'Water':
            cp_water = 4186 #j/kg-K
        elif self.hot_water_loop_fluid == 'GlycolWater30':
            cp_water = 3913 #j/kg-K
        elif self.hot_water_loop_fluid == 'GlycolWater50':
            cp_water = 3558 #j/kg-K
        else:
            cp_water = 4186 #j/kg-K

        HeatRecovered = self.MFR_water * cp_water * (water_out_temp - self.T_water_i) / 1000 #kW
        return HeatRecovered
