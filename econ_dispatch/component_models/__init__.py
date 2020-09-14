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
""".. todo:: Module docstring"""
import abc
import logging
import pkgutil
from copy import deepcopy
from importlib import import_module

LOG = logging.getLogger(__name__)


class ComponentBase(object, metaclass=abc.ABCMeta):
    """Abstract base class for component models

    :param name: name of model
    :param default_parameters: dict of parameter name, value pairs
    :param training_window: period in days over which to train
    :param training_sources: dict of historian topic, name pairs
    :param inputs: dict of message bus topic, name pairs
    :param outputs: dict of name, message bus pairs
    """

    def __init__(
        self,
        name="MISSING_NAME",
        default_parameters={},
        training_window=365,
        training_sources={},
        inputs={},
        outputs={},
    ):
        """Initialize component model"""
        self.name = name
        self.parameters = default_parameters
        self.training_window = int(training_window)
        self.training_sources = training_sources
        self.input_map = inputs
        self.output_map = outputs

    def process_input(self, timestamp, name, value):
        """Override this to process input data from the message bus

        Components will typically want the current state of the device they
        represent as input.

        :param timestamp: time input data was published to the message bus
        :type timestamp: datetime.datetime
        :param name: name of the input from the configuration file
        :param value: value of the input from the message bus
        """
        pass

    def train(self, training_data):
        """Override this to use training data to update parameters

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        pass

    def validate_parameters(self):
        """Return whether parameters exist for this component

        If a more sophisticated method for parameter validation is desired this
        may be overridden

        :returns: whether parameters exist for this component
        :rtype: bool
        """
        return bool(self.parameters)

    def get_mapped_commands(self, optimization_output):
        """Override this to return the new set points on the device based
        on the received component loads and the current state of the component.

        :param optimization_output: full output from optimizer solution
        :type optimization_output: dict
        :returns: map of name, command pairs to be mapped to device topics
        :rtype: dict
        """
        return {}

    def get_commands(self, optimization_output):
        """Process optimizer output into commands and then device topic, set
        point pairs

        :param optimization_output: full output from optimizer solution
        :returns: map of device topic to set point
        :rtype: dict
        """
        mapped_commands = self.get_mapped_commands(optimization_output)
        results = {}
        for output_name, topic in self.output_map.items():
            value = mapped_commands.pop(output_name, None)
            if value is not None:
                results[topic] = value

        for name in mapped_commands:
            LOG.error("NO MAPPED TOPIC FOR {} IN COMPONENT {}. " "DROPPING COMMAND".format(name, self.name))

        return results

    def process_inputs(self, now, inputs):
        """Process data from message bus only if it is in self.input_map

        :param now: time data was published to message bus
        :param inputs: map of name, value pairs
        """
        for topic, input_name in self.input_map.items():
            value = inputs.get(topic)
            if value is not None:
                LOG.debug("{} processing input from topic {}" "".format(self.name, topic))
                self.process_input(now, input_name, value)

    def get_optimization_parameters(self):
        """Get the current parameters of the component for the optimizer

        If something more sophisticated needs to happen this can be overridden

        :returns: map of parameter name to parameter value
        :rtype: dict
        """
        return deepcopy(self.parameters)

    def __str__(self):
        return '"Component: ' + self.name + '"'


COMPONENT_LIST = [x for _, x, _ in pkgutil.iter_modules(__path__)]
COMPONENT_DICT = {}
for COMPONENT_NAME in COMPONENT_LIST:
    try:
        module = import_module(".".join(["econ_dispatch", "component_models", COMPONENT_NAME]))
        klass = module.Component
    except Exception as e:
        LOG.error("Module {name} cannot be imported. Reason: {ex}" "".format(name=COMPONENT_NAME, ex=e))
        continue

    # Validation of algorithm class
    if not issubclass(klass, ComponentBase):
        LOG.warning(
            "The implementation of {name} does not inherit from "
            "econ_dispatch.component_models.ComponentBase."
            "".format(name=COMPONENT_NAME)
        )

    COMPONENT_DICT[COMPONENT_NAME] = klass


def get_component_class(name):
    """Return `Component` class from module named `name`"""
    comp = COMPONENT_DICT.get(name)
    if comp is None:
        LOG.warning("Module {name} not found.".format(name=name))
    return comp
