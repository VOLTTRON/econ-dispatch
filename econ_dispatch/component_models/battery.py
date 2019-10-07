# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2019, Battelle Memorial Institute
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
import logging
import os

import numpy as np

from econ_dispatch.component_models import ComponentBase


LOG = logging.getLogger(__name__)

EXPECTED_PARAMETERS = set(["soc",
                           "charge_eff",
                           "discharge_eff",
                           "min_power",
                           "max_power",
                           "min_soc",
                           "max_soc",
                           "capacity"])

class Component(ComponentBase):
    """Simple battery model

    :param min_power: Unused. Always uses 0
    :param max_power: Maximum (dis)charge. Should use same units as
        power balance
    :param min_soc: Minimum state of charge as fraction of capacity
    :param max_soc: Unused. Always uses 1
    :param capacity: Maximum charge. Should use same units as power balance
    :param ch_point_name: optimization decision variable name
        for charging setpoint
    :param disch_point_name: optimization decision variable name
        for discharging setpoint
    :param kwargs: kwargs for `ComponentBase`
    """
    def __init__(self,
                 min_power=None,
                 max_power=None,
                 min_soc=0.3,
                 max_soc=0.8,
                 capacity=None,
                 ch_point_name="storage_ch_{}_0",
                 disch_point_name="storage_disch_{}_0",
                 **kwargs):
        super(Component, self).__init__(**kwargs)
        self.min_power = float(min_power) # unused
        self.max_power = float(max_power)
        self.min_soc = float(min_soc)
        self.max_soc = float(max_soc) # unused
        self.capacity = capacity
        self.ch_point_name = ch_point_name.format(self.name)
        self.disch_point_name = disch_point_name.format(self.name)
        self.current_soc = None

        self.parameters["min_power"] = self.min_power
        self.parameters["max_power"] = self.max_power
        self.parameters["min_soc"] = self.min_soc
        self.parameters["max_soc"] = self.max_soc
        self.parameters["capacity"] = self.capacity

    def validate_parameters(self):
        """Ensure that at least the necessary parameters are being passed
        to the optimizer (i.e., the model has been trained)"""
        k = set(self.parameters.keys())
        return EXPECTED_PARAMETERS <= k and self.parameters["soc"] is not None

    def get_mapped_commands(self, optimization_output):
        """Return the new set points on the device based on the received
        optimization output and the current state of the component.

        :param optimization_output: full output from optimizer solution
        :type optimization_output: dict
        :returns: map of name, command pairs to be mapped to device topics
        :rtype: dict
        """
        try:
            charge_load = optimization_output[self.ch_point_name]
            discharge_load = optimization_output[self.disch_point_name]
        except KeyError:
            LOG.warning("battery load missing from optimizer output")
            return {}
        return {"charge_load": charge_load, "discharge_load": discharge_load}

    def process_input(self, timestamp, name, value):
        """Process state of charge information from message bus.

        :param timestamp: time input data was published to the message bus
        :type timestamp: datetime.datetime
        :param name: name of the input from the configuration file
        :param value: value of the input from the message bus
        """
        if name == "soc":
            self.current_soc = value
            self.parameters["soc"] = value

    def train(self, training_data):
        """Determine efficiencies from training data

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        charge_eff = self.calculate_charge_eff(training_data, True)
        discharge_eff = self.calculate_charge_eff(training_data, False)

        self.parameters["charge_eff"] = charge_eff
        self.parameters["discharge_eff"] = discharge_eff
        # not updated
        # self.parameters["soc"] = self.current_soc
        # self.parameters["min_power"] = self.min_power
        # self.parameters["max_power"] = self.max_power
        # self.parameters["min_soc"] = self.min_soc
        # self.parameters["max_soc"] = self.max_soc
        # self.parameters["capacity"] = self.capacity

    def calculate_charge_eff(self, charge_training_data, charging):
        """Calculate (dis)charging efficiency from training data

        :param charge_training_data:
        :param charging:
        """
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
        delta_time = delta_time.astype("timedelta64[s]")\
            .astype("float64")/3600.0

        current_power = PowerIn[valid]
        if charging:
            eff = (delta_kWh) / (current_power * delta_time)
        else:
            eff = (current_power * delta_time) / (delta_kWh)

        # Remove garbage values.
        valid_eff = eff[eff<1.0]
        eff_avg = abs(valid_eff.mean())

        LOG.debug("calculate_charge_eff charging {} result {}, dropped {}"
                   " values before calculation".format(
                       charging,eff_avg, len(eff)-len(valid_eff)))

        return eff_avg
