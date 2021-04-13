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

import pandas as pd

from econ_dispatch import utils
from econ_dispatch.component_models import ComponentBase

LOG = logging.getLogger(__name__)


class Component(ComponentBase):
    """Flexible model with a single input/output efficiency curve

    :param input_commodity: input type, such as 'elec', 'gas', or 'heat'
    :param output_commodity: output type, such as 'elec', 'gas', or 'heat'
    :param ramp_up: Maximum positive change in setpoint. Units should match
        internal set point units
    :param ramp_down: Maximum negative change in setpoint. Units should match
        internal setpoint units
    :param min_on: Maximum continuous operation time, in time steps
    :param min_off: Minimum continuous time between operation, in time steps
    :param start_cost: penalty assessed when operation begins
    :param run_cost: penalty assessed each time step while in operation
    :param capacity: maximum set point, irrespective of training data. Units
        should match internal setpoint units
    :param hru_eff: proportion of inefficiency converted to heat
    :param hru_convert: unit conversion factor, [input] = hru_convert*[output]
    :param curve_type: .. todo:: move to "curve_fit_settings" sub-config
    :param report_command: whether to publish unit commitment or raw setpoint
    :param set_point_name: optimization decision variable name for setpoint
    :param preprocess_settings: settings for preprocessor
    :param kwargs: kwargs for `ComponentBase`
    """

    def __init__(
        self,
        input_commodity=None,
        output_commodity=None,
        ramp_up=None,
        ramp_down=None,
        min_on=0,
        min_off=0,
        start_cost=0,
        run_cost=0,
        capacity=None,
        hru_eff=0.0,
        hru_convert=1.0,
        curve_type="poly",
        report_command=True,
        report_set_point=False,
        set_point_name="component_output_{}_0",
        preprocess_settings=None,
        clean_training_data_settings={},
        **kwargs,
    ):
        super(Component, self).__init__(**kwargs)
        self.input_commodity = input_commodity
        self.output_commodity = output_commodity
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.min_on = min_on
        self.min_off = min_off
        self.start_cost = start_cost
        self.run_cost = run_cost
        self.capacity = capacity
        self.hru_eff = hru_eff
        self.hru_convert = hru_convert
        self.output = 0
        self.command_history = [0] * 24
        self.parameters.update(
            {
                "input_commodity": self.input_commodity,
                "output_commodity": self.output_commodity,
                "ramp_up": self.ramp_up,
                "ramp_down": self.ramp_down,
                "min_on": self.min_on,
                "min_off": self.min_off,
                "start_cost": self.start_cost,
                "run_cost": self.run_cost,
                "capacity": self.capacity,
                "hru_eff": self.hru_eff,
                "hru_convert": self.hru_convert,
                "output": self.output,
                "command_history": self.command_history,
            }
        )
        self.curve_type = curve_type
        self.report_command = report_command
        self.report_set_point = report_set_point

        self.set_point_name = set_point_name.format(self.name)

        self.preprocess_settings = preprocess_settings
        self.timestamp_column = clean_training_data_settings.pop("timestamp_column", None)
        self.clean_training_data_settings = clean_training_data_settings

    def get_mapped_commands(self, optimization_output):
        """return the new set points on the device based on the received
        optimization output and the current state of the component.

        :param optimization_output: full output from optimizer solution
        :type optimization_output: dict
        :returns: map of name, command pairs to be mapped to device topics
        :rtype: dict
        """
        try:
            set_point = optimization_output[self.set_point_name]
        except KeyError:
            LOG.error(
                "Required value {} not in optimizer output. "
                "Defaulting {} set point to 0."
                "".format(self.set_point_name, self.name)
            )
            set_point = 0

        command = int(set_point > 0.0)

        self.output = set_point
        self.parameters["output"] = self.output
        self.command_history = self.command_history[1:] + [command]
        self.parameters["command_history"] = self.command_history

        result = {}
        if self.report_command:
            result["command"] = command
        if self.report_set_point:
            result["set_point"] = set_point
        return result

    def train(self, training_data):
        """Fit piecewise-linear efficiency curve parameters to training data

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        training_data = pd.DataFrame(training_data)
        if self.preprocess_settings is not None:
            training_data = utils.preprocess(training_data, **self.preprocess_settings)
        try:
            inputs, outputs = utils.clean_training_data(
                training_data["input"],
                training_data["output"],
                capacity=self.capacity,
                timestamps=training_data.get(self.timestamp_column),
                **self.clean_training_data_settings,
            )
        except ValueError as err:
            LOG.debug("Training data does not meet standards: {}".format(err))
            return

        # LOG.debug([(i, o) for i, o in zip(inputs, outputs)])

        a, b, xmin, xmax = utils.piecewise_linear(inputs, outputs, self.capacity, curve_type=self.curve_type)
        fun_data = {"a": a, "b": b, "min": xmin, "max": xmax}

        LOG.debug(fun_data)
        self.parameters["fundata"] = fun_data
