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

import logging
#import networkx as nx
import datetime
from pprint import pformat
from collections import defaultdict
from econ_dispatch.utils import normalize_training_data, OptimizerCSVOutput
from econ_dispatch.component_models import get_component_class
from econ_dispatch.forecast_models import get_forecast_model_class
from econ_dispatch.optimizer import get_optimization_function

_log = logging.getLogger(__name__)

def build_model_from_config(weather_config,
                            optimizer_config,
                            component_configs,
                            forecast_model_configs,
                            optimization_frequency=60,
                            optimizer_csv_filename=None):
    _log.debug("Starting parse_config")

    weather_type = weather_config["type"]

    module = __import__("weather."+weather_type, globals(), locals(), ['Weather'], 1)
    klass = module.Weather

    weather_model = klass(**weather_config["settings"])

    opt_func = get_optimization_function(optimizer_config)

    optimization_frequency = datetime.timedelta(minutes=optimization_frequency)

    optimizer_csv = None
    if optimizer_csv_filename is not None:
        optimizer_csv = OptimizerCSVOutput(optimizer_csv_filename)

    system_model = SystemModel(opt_func, weather_model, optimization_frequency, optimizer_debug_csv=optimizer_csv)

    for name, config_dict in forecast_model_configs.iteritems():
        model_type = config_dict["type"]
        klass = get_forecast_model_class(name, model_type)
        forecast_model = klass(training_window=config_dict.get("training_window", 365),
                               training_sources=config_dict.get("training_sources", {}),
                               **config_dict.get("settings",{}))

        training_data = config_dict.get("initial_training_data")
        if training_data is not None:
            _log.info("Applying config supplied training data for {} forcast model".format(name))
            training_data = normalize_training_data(training_data)
            forecast_model.train(training_data)

        system_model.add_forecast_model(forecast_model, name)

    for component_dict in component_configs:
        klass_name = component_dict["type"]
        component_name = component_dict["name"]
        klass = get_component_class(klass_name)

        if klass is None:
            _log.error("No component of type: "+klass_name)
            continue

        try:
            component = klass(name=component_name,
                              default_parameters=component_dict.get("default_parameters", {}),
                              training_window=component_dict.get("training_window", 365),
                              training_sources=component_dict.get("training_sources", {}),
                              inputs=component_dict.get("inputs", {}),
                              outputs=component_dict.get("outputs", {}),
                              **component_dict.get("settings", {}))
        except Exception as e:
            _log.exception("Error creating component " + klass_name)
            continue

        training_data = component_dict.get("initial_training_data")
        if training_data is not None:
            _log.info("Applying config supplied training data for {}".format(component_name))
            training_data = normalize_training_data(training_data)
            component.train(training_data)

        if not component.parameters:
            _log.warning("Component {} has no parameters after initialization.".format(component_name))

        system_model.add_component(component, klass_name)

    # connections = config["connections"]
    # for output_component_name, input_component_name in connections:
    #
    #     _log.debug("Adding connection: {} -> {}".format(output_component_name, input_component_name))
    #
    #     try:
    #         if not system_model.add_connection(output_component_name, input_component_name):
    #             _log.error("No compatible outputs/inputs")
    #     except Exception as e:
    #         _log.error("Error adding connection: " + str(e))

    return system_model

class SystemModel(object):
    def __init__(self, optimizer, weather_model, optimization_frequency, optimizer_debug_csv=None):
        #self.component_graph = nx.MultiDiGraph()
        self.instance_map = {}
        self.type_map = defaultdict(dict)

        self.forecast_models = {}

        self.optimizer = optimizer
        self.weather_model = weather_model

        self.optimizer_debug_csv = optimizer_debug_csv

        self.optimization_frequency = optimization_frequency
        self.next_optimization = None

    def add_forecast_model(self, model, name):
        self.forecast_models[name] = model

    def add_component(self, component, type_name):
        #self.component_graph.add_node(component.name, type=type_name)

        self.type_map[type_name][component.name] = component

        if component.name in self.instance_map:
            _log.warning("Duplicate component names: " + component.name)

        self.instance_map[component.name] = component

    def add_connection(self, output_component_name, input_component_name, io_type=None):
        try:
            output_component = self.instance_map[output_component_name]
        except KeyError:
            _log.error("No component named {}".format(output_component_name))
            raise

        try:
            input_component = self.instance_map[input_component_name]
        except KeyError:
            _log.error("No component named {}".format(input_component_name))
            raise

        output_types = output_component.get_output_metadata()
        input_types = input_component.get_input_metadata()

        _log.debug("Output types: {}".format(output_types))
        _log.debug("Input types: {}".format(input_types))

        real_io_types = []
        if io_type is not None:
            real_io_types = [io_type]
        else:
            real_io_types = [x for x in output_types if x in input_types]

        for real_io_type in real_io_types:
            _log.debug("Adding connection for io type: "+real_io_type)
            #self.component_graph.add_edge(output_component.name, input_component.name, label=real_io_type)

        return len(real_io_types)

    def get_forecasts(self, now):
        weather_forecasts = self.weather_model.get_weather_forecast(now)
        #Loads were updated previously when we updated all components
        forecasts = []

        for weather_forecast in weather_forecasts:
            timestamp = weather_forecast.pop("timestamp")
            record = {}
            for name, model in self.forecast_models.iteritems():
                record.update(model.derive_variables(timestamp, weather_forecast))

            forecasts.append(record)

        return forecasts

    def process_inputs(self, now, inputs):
        _log.debug("Updating Components")
        _log.debug("Inputs:\n"+pformat(inputs))
        for component in self.instance_map.itervalues():
            component.process_inputs(now, inputs)

    def run_general_optimizer(self, now, predicted_loads, parameters):
        _log.debug("Running General Optimizer")
        results = self.optimizer(now, predicted_loads, parameters)

        if self.optimizer_debug_csv is not None:
            self.optimizer_debug_csv.writerow(results, predicted_loads, now)

        return results

    def get_parameters(self):
        results= {}

        for type_name, component_dict in self.type_map.iteritems():
            for name, component in component_dict.iteritems():
                parameters = component.get_optimization_parameters()
                try:
                    results[type_name][name] = parameters
                except KeyError:
                    results[type_name] = {name: parameters}

        return results

    def get_commands(self, component_loads):
        _log.debug("Gathering commands")
        result = {}
        for component in self.instance_map.itervalues():
            component_commands = component.get_commands(component_loads)
            for device, commands in component_commands.iteritems():
                result[device] = commands
        return result

    def get_component_input_topics(self):
        results = set()
        for component in self.instance_map.itervalues():
            results.update(component.input_map.keys())

        return results

    def get_training_parameters(self, forecast_models=False):
        results = dict()
        source = self.forecast_models if forecast_models else self.instance_map
        for name, component in source.iteritems():
            # Skip components without training sources configrued.
            if component.training_sources:
                results[name] = (component.training_window, component.training_sources.keys())

        return results

    def apply_all_training_data(self, training_data, forecast_models=False):
        target = self.forecast_models if forecast_models else self.instance_map
        for name, data in training_data.iteritems():
            component = target.get(name)
            if component is None:
                _log.warning("No component named {} to train.".format(name))
                continue
            training_map = component.training_sources
            normalized_data = {}
            for topic , topic_data in data.iteritems():
                mapped_name = training_map.get(topic)
                if mapped_name is not None:
                    normalized = normalize_training_data(topic_data)
                    normalized_data[mapped_name] = normalized
                else:
                    _log.warning("Topic {} has no mapped name for component {}".format(topic, name))
            component.train(normalized_data)

    def invalid_parameters_list(self):
        results = []
        for name, component in self.instance_map.iteritems():
            if not component.validate_parameters():
                results.append(name)
        return results

    def run_optimizer(self, now):
        forecasts = self.get_forecasts(now)
        parameters = self.get_parameters()
        component_loads = self.run_general_optimizer(now, forecasts, parameters)
        _log.debug("Loads: {}".format(pformat(component_loads)))
        commands = self.get_commands(component_loads)

        return commands

    def run(self, now, inputs):
        """Run method for being driven by a script or simulation,
        does own time validation and handles all input."""
        self.process_inputs(now, inputs)

        if self.next_optimization is None:
            self.next_optimization = self.find_starting_datetime(now)

        commands = {}
        if (self.next_optimization <= now):
            _log.info("Running optimizer: " + str(now))
            self.next_optimization = self.next_optimization + self.optimization_frequency
            if self.next_optimization < now:
                # Catch case where we jump way ahead in time
                self.next_optimization = None
            commands = self.validate_run_optimizer(now)
            _log.info("Next optimization: {}".format(self.next_optimization))

        _log.debug("Device commands: {}".format(commands))

        return commands

    def validate_run_optimizer(self, now):
        """For running from a greenlet"""
        commands = {}
        invalid_components = self.invalid_parameters_list()
        if invalid_components:
            _log.error("The following components are unable to provide valid optimization parameters: {}".format(
                invalid_components))
            _log.error("THE OPTIMIZER WILL NOT BE RUN AT THIS TIME.")
        else:
            commands = self.run_optimizer(now)

        return commands

    def find_starting_datetime(self, now):
        """This is taken straight from DriverAgent in MasterDriverAgent."""
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_from_midnight = (now - midnight).total_seconds()
        interval = self.optimization_frequency.total_seconds()

        offset = seconds_from_midnight % interval

        if not offset:
            return now

        previous_in_seconds = seconds_from_midnight - offset
        next_in_seconds = previous_in_seconds + interval

        from_midnight = datetime.timedelta(seconds=next_in_seconds)
        return midnight + from_midnight






