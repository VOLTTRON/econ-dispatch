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

import numpy as np

from econ_dispatch.component_models import ComponentBase
#from econ_dispatch.utils import least_squares_regression
from econ_dispatch import utils

import logging

_log = logging.getLogger(__name__)


DEFAULT_TCHO = 47
DEFAULT_TCDI = 75
DEFAULT_QCH_KW = 500

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


        # Chilled water temperature setpoint outlet from chiller
        # self.Tcho = DEFAULT_TCHO

        # Condenser water temperature inlet temperature to chiller from condenser in F
        # Note that this fixed value of 75F is a placeholder.  We will ultimately
        # need a means of forecasting the condenser water inlet temperature.
        # self.Tcdi = DEFAULT_TCDI

        # building cooling load ASSIGNED TO THIS CHILLER in kW
        # self.Qch_kW = DEFAULT_QCH_KW

        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.start_cost = start_cost
        self.output = 0
        self.max_output = self.capacity

        #self.parameters["cap"] = self.capacity

        # Set to True whenever something happens that causes us to need to recalculate
        # the optimization parameters.
        #self.opt_params_dirty = True
        #self.setup_historical_data()

    # def setup_historical_data(self):
    #     with open(self.training_data_file, 'r') as f:
    #         historical_data = json.load(f)
    #
    #     self.historical_data["P(kW)"] = np.array(historical_data["P(kW)"])
    #     self.historical_data["Qch(tons)"] = np.array(historical_data["Qch(tons)"])

    def get_output_metadata(self):
        return [u"chilled_water"]

    def get_input_metadata(self):
        return [u"electricity"]

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        try:
            run_chiller = component_loads["chiller_x_{}_0".format(self.name)] > 0.0
        except KeyError:
            #Running use case 1.
            run_chiller = component_loads["E_chillerelec_{}_hour00".format(self.name)] > 0.0

        self.output = self.max_output if run_chiller else 0
        self.parameters["output"] = self.output
        return {"command": int(run_chiller)}

    training_inputs_name_map = {
        "outputs": "Qch(tons)",
        "inputs": "P(kW)"
    }

    def train(self, training_data):
        # TODO: Update to calc these from sensor data
        valid = training_data["Qch(tons)"] > 0
        # chiller cooling output in mmBtu/hr (converted from cooling Tons)
        Qch = training_data["Qch(tons)"][valid] * (3.517 / 293.1)

        # chiller power input in kW
        P = training_data["P(kW)"][valid]

        # m_ChillerIGV = least_squares_regression(inputs=Qch, output=P)
        # xmax_ChillerIGV = max(Qch)
        # xmin_ChillerIGV = min(Qch)
        #
        # self.parameters = {
        #     "mat": m_ChillerIGV.tolist(),
        #     "xmax": xmax_ChillerIGV,
        #     "xmin": xmin_ChillerIGV,
        #     "cap": self.capacity
        # }

        _log.debug("X: {}".format(Qch))
        _log.debug("Y: {}".format(P))

        _log.debug("X max: {}".format(max(Qch)))

        a, b, xmin, xmax = utils.piecewise_linear(P, Qch, self.capacity * (3.517 / 293.1))

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

    # def get_optimization_parameters(self):
    #
    #     if not self.opt_params_dirty:
    #         return self.cached_parameters.copy()
    #
    #     # chiller cooling output in mmBtu/hr (converted from cooling Tons)
    #     Qch = np.array(self.historical_data["Qch(tons)"]) * 3.517 / 293.1
    #
    #     # chiller power input in kW
    #     P = self.historical_data["P(kW)"]
    #
    #     m_ChillerIGV = least_squares_regression(inputs=Qch, output=P)
    #     xmax_ChillerIGV = max(Qch)
    #     xmin_ChillerIGV = min(Qch)
    #
    #     self.cached_parameters = {
    #                                 "mat_chillerIGV": m_ChillerIGV.tolist(),
    #                                 "xmax_chillerIGV": xmax_ChillerIGV,
    #                                 "xmin_chillerIGV": xmin_ChillerIGV,
    #                                 "capacity_per_chiller": self.capacity,
    #                                 "chiller_count": self.count
    #                             }
    #     self.opt_params_dirty = False
    #     return self.cached_parameters.copy()


    # def update_parameters(self, timestamp, inputs):
    #     self.Tcho = inputs.get("Tcho", DEFAULT_TCHO)
    #     self.Tcdi = inputs.get("Tcdi", DEFAULT_TCDI)
    #     self.Qch_kW = inputs.get("Qch_kW", DEFAULT_QCH_KW)

    # def predict(self):
    #     # Regression models were built separately (Training Module) and
    #     # therefore regression coefficients are available. Also, forecasted values
    #     # for chiller cooling output were estimated from building load predictions.
    #     # This code is meant to be used for 24 hours ahead predictions.
    #     # The code creates an excel file and writes
    #     # the results on it along with time stamps.
    #
    #     # Gordon-Ng model coefficients
    #     a0, a1, a2, a3 = self.train()
    #
    #     Tcho_K = (self.Tcho - 32) / 1.8 + 273.15#Converting F to Kelvin
    #     Tcdi_K = (self.Tcdi - 32) / 1.8 + 273.15#Converting F to Kelvin
    #
    #     COP = ((Tcho_K / Tcdi_K) - a3 * (self.Qch_kW / Tcdi_K)) / ((a0 + (a1 * (Tcho_K / self.Qch_kW)) + a2 * ((Tcdi_K - Tcho_K) / (Tcdi_K * self.Qch_kW)) + 1)-((Tcho_K / Tcdi_K) - a3 * (self.Qch_kW / Tcdi_K)))
    #     P_Ch_In = self.Qch_kW / COP #Chiller Electric Power Input in kW
    #
    # def train(self):
    #     # This module reads the historical data on temperatures (in Fahrenheit), inlet power to the
    #     # chiller (in kW) and outlet cooling load (in cooling ton) then, converts
    #     # the data to proper units which then will be used for model training. At
    #     # the end, regression coefficients will be written to a file
    #
    #     # data_file = os.path.join(os.path.dirname(__file__), 'CH-Cent-IGV-Historical-Data.json')
    #     with open(self.history_data_file, 'r') as f:
    #         historical_data = json.load(f)
    #
    #     Tcho = historical_data["Tcho(F)"]# chilled water supply temperature in F
    #     Tcdi = historical_data["Tcdi(F)"]# condenser water temperature (outlet from heat rejection and inlet to chiller) in F
    #     Qch = historical_data["Qch(tons)"]# chiller cooling output in Tons of cooling
    #     P = historical_data["P(kW)"]# chiller power input in kW
    #
    #     i = len(Tcho)
    #
    #     # *********************************
    #
    #     COP = np.zeros(i) # Chiller COP
    #     x1 = np.zeros(i)
    #     x2 = np.zeros(i)
    #     x3 = np.zeros(i)
    #     y = np.zeros(i)
    #
    #     for a in range(i):
    #         Tcho[a]= (Tcho[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
    #         Tcdi[a]= (Tcdi[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
    #         COP[a] = float(Qch[a]) / float(P[a])
    #         Qch[a] = Qch[a] * 12000 / 3412 # Converting Tons to kW
    #
    #     for a in range(i):
    #         x1[a] = Tcho[a] / Qch[a]
    #         x2[a] = (Tcdi[a] - Tcho[a]) / (Tcdi[a] * Qch[a])
    #         x3[a] = (((1 / COP[a]) + 1) * Qch[a]) / Tcdi[a]
    #         y[a] = ((((1 / COP[a]) + 1) * Tcho[a]) / Tcdi[a]) - 1
    #
    #     regression_columns = x1, x2, x3
    #     AA = least_squares_regression(inputs=regression_columns, output=y)
    #
    #     return AA
