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
from econ_dispatch import utils
import logging

_log = logging.getLogger(__name__)

DEFAULT_QBP = 55

EXPECTED_PARAMETERS = set(["fundata",
                            "ramp_up",
                            "ramp_down",
                            "start_cost"])

class Component(ComponentBase):
    def __init__(self,
                 ramp_up=None,
                 ramp_down=None,
                 start_cost=None,
                 **kwargs):
        super(Component, self).__init__(**kwargs)

        # Building heating load assigned to Boiler
        self.current_Qbp = 55

        self.max_output = self.capacity
        self.output = 0
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.start_cost = start_cost

        # # Boiler Nameplate parameters (User Inputs)
        # self.Qbprated = 60 #mmBtu/hr
        # self.Gbprated = 90 # mmBtu/hr
        #
        # # NG heat Content 950 Btu/ft3 is assumed
        # self.HC = 0.03355
        #
        # GasInputSubmetering = True #Is metering of gas input to the boilers available? If not, we can't build a regression, and instead will rely on default boiler part load efficiency curves
        # if GasInputSubmetering:
        #     # ********* 5-degree polynomial model coefficients from training*****
        #     self.polynomial_coeffs = self.train()
        # else:
        #     # Use part load curve for 'atmospheric' boiler from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.553.4931&rep=rep1&type=pdf
        #     self.polynomial_coeffs = (0.6978, 3.3745, -15.632, 32.772, -31.45, 11.268)

    def get_output_metadata(self):
        return [u"heated_water"]

    def get_input_metadata(self):
        return [u"natural_gas"]

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        try:
            run_boiler = component_loads["boiler_x_{}_0".format(self.name)]>0.0
        except KeyError:
            #Running use case 1.
            run_boiler = component_loads["Q_boiler_{}_hour00".format(self.name)] > 0.0

        self.output = self.max_output if run_boiler else 0
        self.parameters["output"] = self.output
        return {"command": int(run_boiler)}

    training_inputs_name_map = {
        "outputs": "heat_output",
        "inputs": "gas_input"
    }

    def train(self, training_data):
        valid = training_data["heat_output"] > 0.0

        historical_Qbp = training_data["heat_output"][valid]
        historical_Gbp = training_data["gas_input"][valid]

        _log.debug("X: {}".format(historical_Qbp))
        _log.debug("Y: {}".format(historical_Gbp))

        _log.debug("X max: {}".format(max(historical_Qbp)))

        timestamps = training_data.get("timestamps", None)

        try:
            inputs, outputs = utils.clean_training_data(historical_Gbp, historical_Qbp, self.capacity,
                                                        timestamps=timestamps)
        except ValueError as err:
            _log.debug("Training data does not meet standards: {}".format(err))
            return

        a, b, xmin, xmax = utils.piecewise_linear(inputs, outputs, self.capacity, curve_type='prime_mover')

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
            "output": self.output
        }

        # sort_indexes = np.argsort(historical_Qbp)
        # historical_Qbp = historical_Qbp[sort_indexes]
        # historical_Gbp = historical_Gbp[sort_indexes]
        #
        # n1 = np.nonzero(historical_Qbp < 24)[-1][-1]
        # n2 = np.nonzero(historical_Qbp < 45)[-1][-1]
        #
        # xmin_boiler = np.zeros(3)
        # xmax_boiler = np.zeros(3)
        #
        # m1 = least_squares_regression(inputs=historical_Qbp[  :n1+1], output=historical_Gbp[  :n1+1])
        # m2 = least_squares_regression(inputs=historical_Qbp[n1:n2+1], output=historical_Gbp[n1:n2+1])
        # m3 = least_squares_regression(inputs=historical_Qbp[n2:  ], output=historical_Gbp[n2:  ])
        #
        # mat_boiler = np.array([m1,m2,m3]).T
        #
        # xmax_boiler[0] = max(historical_Qbp[  :n1+1])
        # xmax_boiler[1] = max(historical_Qbp[n1:n2+1])
        # xmax_boiler[2] = max(historical_Qbp[n2:  ])
        #
        # xmin_boiler[0] = min(historical_Qbp[:n1+1])
        # xmin_boiler[1] = xmax_boiler[0]
        # xmin_boiler[2] = xmax_boiler[1]
        #
        # x1 = xmin_boiler[0]
        # x2 = xmin_boiler[2]
        #
        # y1 = mat_boiler[0][0] + mat_boiler[1][0] * xmax_boiler[0]
        # y2 = mat_boiler[0][2] + mat_boiler[1][2] * xmin_boiler[2]
        #
        # mat_boiler[1][1] = (y2 - y1) / (x2 - x1)
        # mat_boiler[0][1] = y1 - mat_boiler[1][1] * x1
        #
        # self.parameters = {
        #                         "xmin": xmin_boiler.tolist(),
        #                         "xmax": xmax_boiler.tolist(),
        #                         "mat": mat_boiler.tolist(),
        #                         "cap": self.capacity
        #                   }

    # def update_parameters(self, timestamp, inputs):
    #     self.current_Qbp = inputs.get("Qbp", DEFAULT_QBP)

    # def get_optimization_parameters(self):
    #
    #     if not self.opt_params_dirty:
    #         return self.cached_parameters.copy()
    #
    #     historical_Qbp = self.historical_data["boiler_heat_output"]
    #     historical_Gbp = self.historical_data["boiler_gas_input"]
    #
    #     sort_indexes = np.argsort(historical_Qbp)
    #     historical_Qbp = historical_Qbp[sort_indexes]
    #     historical_Gbp = historical_Gbp[sort_indexes]
    #
    #     n1 = np.nonzero(historical_Qbp < 24)[-1][-1]
    #     n2 = np.nonzero(historical_Qbp < 45)[-1][-1]
    #
    #     xmin_boiler = np.zeros(3)
    #     xmax_boiler = np.zeros(3)
    #
    #     m1 = least_squares_regression(inputs=historical_Qbp[:n1 + 1], output=historical_Gbp[:n1 + 1])
    #     m2 = least_squares_regression(inputs=historical_Qbp[n1:n2 + 1], output=historical_Gbp[n1:n2 + 1])
    #     m3 = least_squares_regression(inputs=historical_Qbp[n2:], output=historical_Gbp[n2:])
    #
    #     mat_boiler = np.array([m1, m2, m3]).T
    #
    #     xmax_boiler[0] = max(historical_Qbp[:n1 + 1])
    #     xmax_boiler[1] = max(historical_Qbp[n1:n2 + 1])
    #     xmax_boiler[2] = max(historical_Qbp[n2:])
    #
    #     xmin_boiler[0] = min(historical_Qbp[:n1 + 1])
    #     xmin_boiler[1] = xmax_boiler[0]
    #     xmin_boiler[2] = xmax_boiler[1]
    #
    #     x1 = xmax_boiler[0]
    #     x2 = xmin_boiler[2]
    #     y1 = mat_boiler[0][0] + mat_boiler[1][0] * xmax_boiler[0]
    #     y2 = mat_boiler[0][2] + mat_boiler[1][2] * xmin_boiler[2]
    #
    #     mat_boiler[1][1] = (y2 - y1) / (x2 - x1)
    #     mat_boiler[0][1] = y1 - mat_boiler[1][1] * x1
    #
    #     self.cached_parameters = {
    #         "xmin_boiler": xmin_boiler.tolist(),
    #         "xmax_boiler": xmax_boiler.tolist(),
    #         "mat_boiler": mat_boiler.tolist(),
    #         "cap_boiler": self.capacity
    #     }
    #     self.opt_params_dirty = False
    #     return self.cached_parameters.copy()



    # def predict(self):
    #     a0, a1, a2, a3, a4, a5 = self.polynomial_coeffs
    #     if self.current_Qbp > self.Qbprated:
    #         Qbp = self.Qbprated
    #     else:
    #         Qbp = self.current_Qbp
    #
    #     xbp = Qbp / self.Qbprated # part load ratio
    #     ybp = a0 + a1*xbp + a2*(xbp)**2 + a3*(xbp)**3 + a4*(xbp)**4 + a5*(xbp)**5# relative efficiency (multiplier to ratred efficiency)
    #     Gbp = (Qbp * self.Gbprated) / (ybp * self.Qbprated)# boiler gas heat input in mmBtu
    #     FC = Gbp / self.HC #fuel consumption in cubic meters per hour
    #
    # def train(self):
    #     # This module reads the historical data on boiler heat output and
    #     # gas heat input both in mmBTU/hr then, converts
    #     # the data to proper units which then will be used for model training.
    #
    #     # boiler gas input in mmBTU
    #     # Note from Nick Fernandez: Most sites will not have metering for gas inlet
    #     # to the boiler.  I'm creating a second option to use a defualt boiler
    #     # curve
    #     historical_Gbp = self.historical_data["boiler_gas_input"]
    #
    #     # boiler heat output in mmBTU
    #     historical_Qbp = self.historical_data["boiler_heat_output"]
    #
    #     i = len(historical_Gbp)
    #
    #     # ****** Static Inputs (Rating Condition + Natural Gas Heat Content *******
    #     Qbprated = 60.0 #boiler heat output at rated condition - user input (mmBtu)
    #     Gbprated = 90.0 #boiler gas heat input at rated condition - user input (mmBtu)
    #     #**************************************************************************
    #
    #
    #
    #     xbp = historical_Qbp / Qbprated
    #     xbp2 = xbp ** 2
    #     xbp3 = xbp ** 3
    #     xbp4 = xbp ** 4
    #     xbp5 = xbp ** 5
    #     ybp = (historical_Qbp / historical_Gbp) / (Qbprated / Gbprated)
    #
    #     # xbp = np.zeros(i)
    #     # xbp2 = np.zeros(i)
    #     # xbp3 = np.zeros(i)
    #     # xbp4 = np.zeros(i)
    #     # xbp5 = np.zeros(i)
    #     # ybp = np.zeros(i)
    #     #
    #     # for a in range(i):
    #     #     xbp[a] = historical_Qbp[a] / Qbprated
    #     #     xbp2[a] = xbp[a]**2
    #     #     xbp3[a] = xbp[a]**3
    #     #     xbp4[a] = xbp[a]**4
    #     #     xbp5[a] = xbp[a]**5
    #     #     ybp[a] = (historical_Qbp[a] / historical_Gbp[a]) / (float(Qbprated) / float(Gbprated))
    #
    #     regression_columns = xbp, xbp2, xbp3, xbp4, xbp5
    #     AA = least_squares_regression(inputs=regression_columns, output=ybp)
    #     return AA
