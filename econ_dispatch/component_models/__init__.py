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

import abc
import logging
import pkgutil
from copy import deepcopy

_log = logging.getLogger(__name__)

_componentList = [x for _, x, _ in pkgutil.iter_modules(__path__)]

_componentDict = {}

valid_io_types = set([u"heated_water",
                      u"heated_air",
                      u"waste_heat",
                      u"heat",
                      u"chilled_water",
                      u"chilled_air",
                      u"electricity",
                      u"natural_gas",
                      u"steam"])

class ComponentBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name="MISSING_NAME",
                 default_parameters={},
                 training_window=365,
                 training_sources={},
                 inputs={},
                 outputs={},
                 cop=None,
                 efficiency=None,
                 capacity=None):

        self.name = name
        self.parameters = default_parameters
        self.training_window = int(training_window)
        self.training_sources = training_sources
        self.input_map = inputs
        self.output_map = outputs
        self.eff_cop = efficiency if efficiency is not None else cop
        self.capacity = capacity

    training_inputs_name_map = {
        "outputs": "outputs",
        "inputs": "inputs"
    }

    @classmethod
    def rename_default_curve_data(cls, training_data):
        results = {}
        results[cls.training_inputs_name_map["outputs"]] = training_data["outputs"]
        results[cls.training_inputs_name_map["inputs"]] = training_data["inputs"]
        return results


    def get_input_metadata(self):
        """Must return a string describing the required input for this component.
        This is used by the model validator to determine if the configured network is valid.

        This function must return a list of strings that names it's input type.
        e.g. "chilled_water"

        If the component has no input return None.

        If the component has more than one output return a list of names.

        This is used during validation of the model after/during configuration.
        """
        return []

    def get_output_metadata(self):
        """Must return a string describing the output for this component.
        This is used by the model validator to determine if the configured network is valid.

        This function must return a list of strings that names it's output type.
        e.g. "chilled_water"

        If the component has no output return None.

        If the component has more than one output return a list of names.

        This is used during validation of the model after/during configuration.
        """
        return []

    def get_mapped_commands(self, component_loads):
        """Override this to return the new set points on the device based
        on the received component loads and the current state of the component.
        Return values must take the form:

        {"output1": value1,
         "output2": value2}
        """
        return {}

    def process_input(self, timestamp, name, value):
        """Override this to process input data from the platform.
        Components will typically want the current state of the device they
        represent as input.
        name - Name of the input from the configuration file.
        value - value of the input from the message bus.
        """
        pass

    def train(self, training_data):
        """Override this to use training data to update parameters
        training_data takes the form:

        {
         "input_name1": [value1, value2,...],
         "input_name2": [value1, value2,...]
        }
        """
        pass

    def validate_parameters(self):
        """Returns true if parameters exist for this component. False otherwise.

        If more a sophisticated method for parameter validation is desired this
        may be overridden.
        """
        return bool(self.parameters)

    def get_commands(self, component_loads):
        """Returns the commands for this component mapped to the topics specified
        in the configuration file."""
        mapped_commands = self.get_mapped_commands(component_loads)
        results = {}
        for output_name, topic in self.output_map.iteritems():
            value = mapped_commands.pop(output_name, None)
            if value is not None:
                results[topic] = value

        for name in mapped_commands:
            _log.error("NO MAPPED TOPIC FOR {}. DROPPING COMMAND".format(name))

        return results

    def process_inputs(self, now, inputs):
        for topic, input_name in self.input_map.iteritems():
            value = inputs.get(topic)
            if value is not None:
                _log.debug("{} processing input from topic {}".format(self.name, topic))
                self.process_input(now, input_name, value)

    def get_optimization_parameters(self):
        """Get the current parameters of the component for the optimizer.
        Returned values must take the form of a dictionary.

        If something more sophisticated needs to happen this can be overridden."""
        return deepcopy(self.parameters)


    def __str__(self):
        return '"Component: ' + self.name + '"'

for componentName in _componentList:
    try:
        module = __import__(componentName, globals(), locals(), ['Component'], 1)
        klass = module.Component
    except Exception as e:
        _log.error('Module {name} cannot be imported. Reason: {ex}'.format(name=componentName, ex=e))
        continue

    #Validation of Algorithm class

    if not issubclass(klass, ComponentBase):
        _log.warning('The implementation of {name} does not inherit from econ_dispatch.component_models.ComponentBase.'.format(name=componentName))

    _componentDict[componentName] = klass


def get_component_class(name):
    return _componentDict.get(name)