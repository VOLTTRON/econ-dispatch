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

from econ_dispatch.component_models import ComponentBase
from econ_dispatch.utils import least_squares_regression

DEFAULT_TCHO = 44
DEFAULT_TCDI = 75
DEFAULT_QCH_KW = 1758.5


class Component(ComponentBase):
    def __init__(self, **kwargs):
        super(Component, self).__init__(**kwargs)
        # Regression models were built separately (Training Module) and
        # therefore regression coefficients are available. Also, forecasted values
        # for Chiller cooling output were estimated from building load predictions.
        # This code is meant to be used for 24 hours ahead predictions.
        # The code creates an excel file and writes
        # the results on it along with time stamps
    
        # Chilled water temperature setpoint outlet from chiller
        self.Tcho = DEFAULT_TCHO

        # Condenser water temperature inlet temperature to chiller from condenser in F
        # Note that this fixed value of 75F is a placeholder.  We will ultimately
        # need a means of forecasting the condenser water inlet temperature.
        self.Tcdi = DEFAULT_TCDI

        # building cooling load ASSIGNED TO THIS CHILLER in kW
        self.Qch_kW = DEFAULT_QCH_KW

    def get_output_metadata(self):
        return ""

    def get_input_metadata(self):
        return ""

    def get_optimization_parameters(self):
        self.predict()
        return {}

    def update_parameters(self, Tcho=DEFAULT_TCHO,
                          Tcdi=DEFAULT_TCDI,
                          Qch_kW=DEFAULT_QCH_KW):
        self.Tcho = Tcho
        self.Tcdi = Tcdi
        self.Qch_kW = Qch_kW

    def predict(self):
        # Gordon-Ng model coefficients
        Qchmax, (a0, a1, a2, a3, a4) = self.train()
    
        Tcho_K = (self.Tcho - 32) / 1.8 + 273.15#Converting F to Kelvin
        Tcdi_K = (self.Tcdi - 32) / 1.8 + 273.15#Converting F to Kelvin
        Qch_kW = self.Qch_kW
    
        COP = ((Tcho_K / Tcdi_K) - a4 * (Qch_kW / Tcdi_K)) / ((a0 + (a1 + a2 * (Qch_kW / Qchmax)) * (Tcho_K / Qch_kW) + a3 * ((Tcdi_K - Tcho_K) / (Tcdi_K * Qch_kW)) + 1) - ((Tcho_K / Tcdi_K) - a4 * (Qch_kW / Tcdi_K)))
        #Coefficient of Performance(COP) of chiller from regression
    
        P_Ch_In = Qch_kW / COP #Chiller Electric Power Input in kW
    
    def train(self):
        # This module reads the historical data on temperatures (in Fahrenheit), inlet power to the
        # chiller (in kW) and outlet cooling load (in cooling ton) then, converts
        # the data to proper units which then will be used for model training. At
        # the end, regression coefficients will be written to an excel file

        data_file = os.path.join(os.path.dirname(__file__), 'CH-Cent-VSD-Historical-Data.json')
        with open(data_file, 'r') as f:
            historical_data = json.load(f)
    
        Tcho = historical_data["Tcho(F)"]# chilled water supply temperature in F
        Tcdi = historical_data["Tcdi(F)"]# condenser water temperature (outlet from heat rejection and inlet to chiller) in F
        Qch = historical_data["Qch(tons)"]# chiller cooling output in Tons of cooling
        P = historical_data["P(kW)"]# chiller power input in kW
        
        i = len(Tcho)
        U = np.ones(i)
    
        NameplateAvailable = True #User input of maximum chiller capacity, if available
        if NameplateAvailable:
            Qchmax_Tons= 500  #Chiller capacity in cooling tons
            Qchmax = Qchmax_Tons*12000/3412
        else:
            Qchmax = max(Qch)
    
        # *********************************
    
        COP = np.zeros(i) # Chiller COP
        x1 = np.zeros(i)
        x2 = np.zeros(i)
        x3 = np.zeros(i)
        x4 = np.zeros(i)
        y = np.zeros(i)
    
        for a in range(i):
            Tcho[a] = (Tcho[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
            Tcdi[a] = (Tcdi[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
            Qch[a] = Qch[a] * 12000 / 3412 # Converting tons to kW
            COP[a] = Qch[a] / P[a]
    
        for a in range(i):
            x1[a] = Tcho[a] / Qch[a]
            x2[a] = Tcho[a] / Qchmax
            x3[a] = (Tcdi[a] - Tcho[a]) / (Tcdi[a] * Qch[a])
            x4[a] = (((1 / COP[a]) + 1) * Qch[a]) / Tcdi[a]
            y[a] = ((((1 / COP[a]) + 1) * Tcho[a]) / Tcdi[a]) - 1

        regression_columns = x1, x2, x3, x4
        AA = least_squares_regression(inputs=regression_columns, output=y)
    
        return Qchmax, AA
    
