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

_componentList = [name for _, name, _ in pkgutil.iter_modules(__path__)]

_componentDict = {}

valid_io_types = set([u"heated_water",
                      u"heated_air",
                      u"waste_heat",
                      u"heat",
                      u"chilled_water",
                      u"chilled_air",
                      u"electricity",
                      u"natural_gas"])

class ComponentBase(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, name="MISSING_NAME", **kwargs):
        self.update_parameters(None, **kwargs)
        self.name = name

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

    def get_commands(self, component_loads):
        """Get the set points for a component based on the optimized component load
        and the current state of the component.
        Return values must take the form:

        {"device1": {"command1": 50.0, "command2": True},
         "device2": {"command3": 22.0}}

        Typically a component will only provide command for a single device.
        """
        return {}

    @abc.abstractmethod
    def get_optimization_parameters(self):
        """Get the current parameters of the component for the optimizer.
        Returned values must take the form of a dictionary."""
        pass

    @abc.abstractmethod
    def update_parameters(self, timestamp, **kwargs):
        """Update the internal parameters of the component based on the input values."""
        pass

    def __str__(self):
        return '"Component: ' + self.name + '"'

for componentName in _componentList:
    try:
        module = __import__(componentName,globals(),locals(),['Component'], 1)
        klass = module.Component
    except Exception as e:
        logging.error('Module {name} cannot be imported. Reason: {ex}'.format(name=componentName, ex=e))
        continue

    #Validation of Algorithm class

    if not issubclass(klass, ComponentBase):
        logging.warning('The implementation of {name} does not inherit from econ_dispatch.component_models.ComponentBase.'.format(name=componentName))

    _componentDict[componentName] = klass


def get_component_class(name):
    return _componentDict.get(name)