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

DEFAULT_QBP = 55

class Component(ComponentBase):
    def __init__(self, **kwargs):
        super(Component, self).__init__(**kwargs)

        # Building heating load assigned to Boiler
        self.Qbp = 55

        # Boiler Nameplate parameters (User Inputs)
        self.Qbprated = 60 #mmBtu/hr
        self.Gbprated = 90 # mmBtu/hr

        # NG heat Content 950 Btu/ft3 is assumed
        self.HC = 0.03355 

        GasInputSubmetering = True #Is metering of gas input to the boilers available? If not, we can't build a regression, and instead will rely on default boiler part load efficiency curves
        if GasInputSubmetering:
            # ********* 5-degree polynomial model coefficients from training*****
            self.polynomial_coeffs = self.train()
        else:
            # Use part load curve for 'atmospheric' boiler from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.553.4931&rep=rep1&type=pdf
            self.polynomial_coeffs = (0.6978, 3.3745, -15.632, 32.772, -31.45, 11.268)

    def get_output_metadata(self):
        return [u"heated_water"]

    def get_input_metadata(self):
        return [u"natural_gas"]

    def get_optimization_parameters(self):
        self.predict()
        return {}

    def update_parameters(self, Qbp=DEFAULT_QBP):
        self.Qbp = Qbp

    def predict(self):
        a0, a1, a2, a3, a4, a5 = self.polynomial_coeffs
        if Qbp > self.Qbprated:
            Qbp = self.Qbprated
        else:
            Qbp = self.Qbp

        xbp = Qbp / self.Qbprated # part load ratio
        ybp = a0 + a1*xbp + a2*(xbp)**2 + a3*(xbp)**3 + a4*(xbp)**4 + a5*(xbp)**5# relative efficiency (multiplier to ratred efficiency)
        Gbp = (Qbp * self.Gbprated) / (ybp * self.Qbprated)# boiler gas heat input in mmBtu
        FC = Gbp / self.HC #fuel consumption in cubic meters per hour

    def train(self):
        # This module reads the historical data on boiler heat output and
        # gas heat input both in mmBTU/hr then, converts
        # the data to proper units which then will be used for model training. At
        # the end, regression coefficients will be written to a file

        with open("Boiler-Historical-Data.json", 'r') as f:
            historical_data = json.load(f)

        # boiler gas input in mmBTU
        # Note from Nick Fernandez: Most sites will not have metering for gas inlet
        # to the boiler.  I'm creating a second option to use a defualt boiler
        # curve
        Gbp = historical_data["boiler_gas_input"]

        # boiler heat output in mmBTU
        Qbp = historical_data["boiler_heat_output"]

        i = len(Gbp)
        U = np.ones(i)

        # ****** Static Inputs (Rating Condition + Natural Gas Heat Content *******
        Qbprated = 60 #boiler heat output at rated condition - user input (mmBtu)
        Gbprated = 90 #boiler gas heat input at rated condition - user input (mmBtu)
        #**************************************************************************

        xbp = np.zeros(i)
        xbp2 = np.zeros(i)
        xbp3 = np.zeros(i)
        xbp4 = np.zeros(i)
        xbp5 = np.zeros(i)
        ybp = np.zeros(i)

        for a in range(i):
            xbp[a] = Qbp[a] / Qbprated
            xbp2[a] = xbp[a]**2
            xbp3[a] = xbp[a]**3
            xbp4[a] = xbp[a]**4
            xbp5[a] = xbp[a]**5
            ybp[a] = (Qbp[a] / Gbp[a]) / (float(Qbprated) / float(Gbprated))

        #*******Multiple Linear Regression***********
        XX = np.column_stack((U, xbp, xbp2, xbp3, xbp4, xbp5))#matrix of predictors
        AA, resid, rank, s = np.linalg.lstsq(XX, ybp)
        #********************************************

        return AA

