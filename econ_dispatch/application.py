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

from system_model import SystemModel
from econ_dispatch.component_models import get_component_class
from econ_dispatch.forecast_models import get_forecast_model_class
from econ_dispatch.optimizer import get_optimization_function
from econ_dispatch.utils import OptimizerCSVOutput
from collections import OrderedDict, defaultdict
import datetime
import logging
_log = logging.getLogger(__name__)


#Copied from driven framework to use until we are integrated with VOLTTRON.
class Results(object):
    def __init__(self, terminate=False):
        self.commands = OrderedDict()
        self.devices = OrderedDict()
        self.log_messages = []
        self._terminate = terminate
        self.table_output = defaultdict(list)

    def command(self, point, value, device=None):
        if device is None:
            self.commands[point] = value
        else:
            if device not in self.devices.keys():
                self.devices[device] = OrderedDict()
            self.devices[device][point] = value
        if self.devices is None:
            self.commands[point]=value
        else:
            if  device not in self.devices.keys():
                self.devices[device] = OrderedDict()
            self.devices[device][point]=value

    def log(self, message, level=logging.DEBUG):
        self.log_messages.append((level, message))

    def terminate(self, terminate):
        self._terminate = bool(terminate)

    def insert_table_row(self, table, row):
        self.table_output[table].append(row)

def build_model_from_config(config):
    _log.debug("Starting parse_config")

    weather_config = config["weather"]

    weather_type = weather_config["type"]

    module = __import__("weather."+weather_type, globals(), locals(), ['Weather'], 1)
    klass = module.Weather

    weather_model = klass(**weather_config["settings"])

    opt_func = get_optimization_function(config["optimizer"])

    optimization_frequency = int(config.get("optimization_frequency", 60))

    optimization_frequency = datetime.timedelta(minutes=optimization_frequency)

    optimizer_csv = None
    optimizer_csv_filename = config.get("optimizer_debug")
    if optimizer_csv_filename is not None:
        optimizer_csv = OptimizerCSVOutput(optimizer_csv_filename)

    system_model = SystemModel(opt_func, weather_model, optimization_frequency, optimizer_debug_csv=optimizer_csv)

    forecast_model_configs = config["forecast_models"]
    components = config["components"]
    connections = config["connections"]

    for name, config_dict in forecast_model_configs.iteritems():
        model_type = config_dict["type"]
        forecast_model = get_forecast_model_class(name, model_type, **config_dict.get("settings",{}))
        system_model.add_forecast_model(forecast_model, name)

    for component_dict in components:
        klass_name = component_dict["type"]
        component_name = component_dict["name"]
        klass = get_component_class(klass_name)

        if klass is None:
            _log.error("No component of type: "+klass_name)
            continue

        try:
            component = klass(name=component_name, **component_dict.get("settings", {}))
        except Exception as e:
            _log.exception("Error creating component " + klass_name)
            continue

        system_model.add_component(component, klass_name)

    for output_component_name, input_component_name in connections:

        _log.debug("Adding connection: {} -> {}".format(output_component_name, input_component_name))

        try:
            if not system_model.add_connection(output_component_name, input_component_name):
                _log.error("No compatible outputs/inputs")
        except Exception as e:
            _log.error("Error adding connection: " + str(e))


    return system_model

class Application(object):
    def __init__(self, model_config={}, **kwargs):
        super(Application, self).__init__(**kwargs)
        self.model = build_model_from_config(model_config)

    @classmethod
    def output_format(cls):
        """Not needed on VOLTTRON platform"""
        return {}

    def run(self, time, inputs):
        device_commands = self.model.run(time, inputs)
        results = Results()
        for device, commands in device_commands.iteritems():
            for point, value in commands.iteritems():
                results.command(point, value, device)

        return results

