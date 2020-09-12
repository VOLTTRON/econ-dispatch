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
from collections import defaultdict
import datetime
from importlib import import_module
import logging
from pprint import pformat

from econ_dispatch.utils import (normalize_training_data,
                                 OptimizerCSVOutput,
                                 PiecewiseError,
                                 get_default_curve)
from econ_dispatch.component_models import get_component_class
from econ_dispatch.forecast_models import get_forecast_class
from econ_dispatch.optimizer import get_optimization_function

LOG = logging.getLogger(__name__)


def build_model_from_config(weather_config,
                            optimizer_config,
                            component_configs,
                            forecast_configs,
                            optimizer_csv_filename=None,
                            command_csv_filename=None):
    """Initialize system model from configs

    :param weather_config: weather model configuration
    :param optimizer_config: optimization configuration
    :param component_configs: dict of component model configurations
    :param forecast_configs: dict of forecast model configurations
    :param optimizer_csv_filename: path to write optimizer debug CSV
    :param command_csv_filename: path to write command debug CSV
    """
    LOG.debug("Starting parse_config")

    weather_type = weather_config["type"]
    module = import_module(".".join(["econ_dispatch", "weather", weather_type]))
    klass = module.Weather
    weather_model = klass(**weather_config.get("settings", {}))
    training_data = weather_config.get("initial_training_data")
    if training_data is not None:
        LOG.info("Applying config supplied training data for "
                "weather forcast model")
        training_data = normalize_training_data(training_data)
        try:
            weather_model.train(training_data)
        except PiecewiseError:
            pass

    opt_func = get_optimization_function(optimizer_config)

    if optimizer_csv_filename is None:
        optimizer_csv = None
    else:
        optimizer_csv = OptimizerCSVOutput(optimizer_csv_filename)

    if command_csv_filename is None:
        command_csv = None
    else:
        command_csv = OptimizerCSVOutput(command_csv_filename)

    system_model = SystemModel(opt_func,
                               weather_model,
                               optimizer_debug_csv=optimizer_csv,
                               command_debug_csv=command_csv)

    for config_dict in forecast_configs:
        name = config_dict["name"]
        klass_name = config_dict["type"]
        klass = get_forecast_class(klass_name)

        if klass is None:
            LOG.error("No component of type: "+klass_name)
            continue

        forecast_model = klass(training_window=config_dict.get("training_window", 365),
                               training_sources=config_dict.get("training_sources", {}),
                               **config_dict.get("settings", {}))

        training_data = config_dict.get("initial_training_data")
        if training_data is not None:
            LOG.info("Applying config supplied training data for "
                     "{} forcast model".format(name))
            training_data = normalize_training_data(training_data)
            try:
                forecast_model.train(training_data)
            except PiecewiseError:
                pass

        system_model.add_forecast_model(forecast_model, name)

    for config_dict in component_configs:
        name = config_dict["name"]
        klass_name = config_dict["type"]
        klass = get_component_class(klass_name)

        if klass is None:
            LOG.error("No component of type: "+klass_name)
            continue

        try:
            component = klass(
                name=name,
                default_parameters=config_dict.get("default_parameters", {}),
                training_window=config_dict.get("training_window", 365),
                training_sources=config_dict.get("training_sources", {}),
                inputs=config_dict.get("inputs", {}),
                outputs=config_dict.get("outputs", {}),
                **config_dict.get("settings", {}))
        except Exception as e:
            LOG.exception("Error creating component {}".format(klass_name))
            continue

        training_data = config_dict.get("initial_training_data")
        if training_data is not None:
            LOG.info("Applying config supplied training data for {}".format(
                name))
            training_data = normalize_training_data(training_data)
            try:
                component.train(training_data)
            except Exception as e:
                LOG.warning("Failed to train component {} with "
                            "initial_training_data. Using default curve."
                            "".format(name))
                LOG.warning("Exception raised by train function: {}".format(repr(e)))

        if not component.parameters:
            LOG.warning("Component %s has no parameters after initialization.",
                        name)

        system_model.add_component(component, klass_name)

    return system_model

class SystemModel(object):
    """Coordinates data flow between forecasts, components, optimizer,
    and agent

    :param optimizer: optimization
    :type optimizer: function, defined as `_optimize` in econ_dispatch.optimizer
    :param weather_model: weather model
    :type weather_model: child of econ_dispatch.forecast_models.ForecastBase
    :param optimizer_csv_filename: path to write optimizer debug CSV
    :param command_debug_csv: path to write command debug CSV
    """
    def __init__(self,
                 optimizer,
                 weather_model,
                 optimizer_debug_csv=None,
                 command_debug_csv=None):
        self.optimizer = optimizer
        self.weather_model = weather_model

        # components
        self.instance_map = {}
        self.type_map = defaultdict(dict)
        # forecasts
        self.forecast_models = {}

        self.optimizer_debug_csv = optimizer_debug_csv
        self.command_debug_csv = command_debug_csv

    def add_forecast_model(self, forecast, name):
        """Add forecast model to system

        :param forecast: forecast model to add
        :type forecast: child of econ_dispatch.forecast_models.ForecastBase
        :name: unique name of forecast
        """
        if name in self.forecast_models:
            LOG.warning("Duplicate forecast names: " + name)

        self.forecast_models[name] = forecast

    def add_component(self, component, type_name):
        """Add component model to system

        :param component: component model to add
        :type component: child of econ_dispatch.component_models.ComponentBase
        :param type_name: type of component
        """
        self.type_map[type_name][component.name] = component

        if component.name in self.instance_map:
            LOG.warning("Duplicate component names: " + component.name)
        self.instance_map[component.name] = component

    def get_forecasts(self, now):
        """Query each forecast model with each weather forecast

        :param now: time to start forecasts from
        :type now: datetime.datetime
        :returns: a forecast of every type for each weather forecast
        :rtype: list of dicts
        """
        weather_forecasts = self.weather_model.get_weather_forecast(now)

        forecasts = []
        for weather_forecast in weather_forecasts:
            timestamp = weather_forecast.pop("timestamp")
            record = {"timestamp": timestamp}
            for model in self.forecast_models.values():
                record.update(model.derive_variables(timestamp, weather_forecast))

            forecasts.append(record)
        return forecasts

    def get_parameters(self):
        """Query each component for its optimization parameters

        :returns: component parameters organized by type then by name
        :rtype: dict of dicts
        """
        results = {}
        for type_name, component_dict in self.type_map.items():
            for name, component in component_dict.items():
                parameters = component.get_optimization_parameters()
                try:
                    if results[type_name].get(name) is not None:
                        LOG.warning("Multiple components with name {name} "
                                    "of type {type}. Overwriting parameters"
                                    "".format(name=name, type=type_name))
                    results[type_name][name] = parameters
                except KeyError:
                    results[type_name] = {name: parameters}

        return results

    def get_component_input_topics(self):
        """Gather message bus topics to follow from components

        :returns: message bus topics to follow
        :rtype: set
        """
        results = set()
        for component in self.instance_map.values():
            results.update(list(component.input_map.keys()))

        return results

    def process_inputs(self, now, inputs):
        """Pass input data from message bus to each component for
        further processing"""
        LOG.debug("Updating components with inputs: "+pformat(inputs))
        for component in self.instance_map.values():
            component.process_inputs(now, inputs)

    def get_training_parameters(self, forecast_models=False):
        """Gather time windows and historian topics to query for training data

        :param forecast_models: whether to query forecast or component models
        :type forecast_models: bool
        :returns: mapping of model name to time window and historian topics
        :rtype: dict of tuples
        """
        results = dict()
        source = self.forecast_models if forecast_models else self.instance_map
        for name, component in source.items():
            # Skip components without training sources configured.
            if component.training_sources:
                results[name] = (component.training_window,
                                 list(component.training_sources.keys()))

        return results

    def apply_all_training_data(self, training_data, forecast_models=False):
        """Train models on data from historian

        :param training_data: data organized by model name then historian topic
        :type training_data: dict of dicts
        :param forecast_models: whether to query forecast or component models
        :type forecast_models: bool
        """
        target = self.forecast_models if forecast_models else self.instance_map
        for name, data in training_data.items():
            component = target.get(name)
            if component is None:
                LOG.warning("No component named {} to train.".format(name))
                continue
            # map historian topic to component's expected training data headers
            training_map = component.training_sources
            normalized_data = {}
            for topic, topic_data in data.items():
                mapped_name = training_map.get(topic)
                if mapped_name is not None:
                    normalized = normalize_training_data(topic_data)
                    normalized_data[mapped_name] = normalized
                else:
                    LOG.warning("Topic {} has no mapped name for component {}"
                                "".format(topic, name))
            component.train(normalized_data)

    def invalid_parameters_list(self):
        """Return list of components with invalid parameters"""
        results = []
        for name, component in self.instance_map.items():
            if not component.validate_parameters():
                results.append(name)
        return results

    def get_commands(self, optimization_results):
        """Pass optimization results to components and get back commands

        :param optimization_results: full results from optimization solution
        :returns: device commands defined by components
        :rtype: dict
        """
        LOG.debug("Gathering commands")
        result = {}
        for component in self.instance_map.values():
            component_commands = component.get_commands(optimization_results)
            for device, commands in component_commands.items():
                if device in result:
                    LOG.warning("Command to device {} being overwritten by {}"
                                "".format(device, component.name))
                result[device] = commands
        return result

    def run_optimizer(self, now):
        """Validate component parameters, gather forecasts and component
        parameters, run optimizer, then process results into commands

        :param now: start of optimization window
        :type now: datetime.datetime
        :returns: dict of device name: command pairs
        """
        invalid_components = self.invalid_parameters_list()
        if invalid_components:
            LOG.error("The following components are unable to provide valid "
                      "optimization parameters: {}".format(invalid_components))
            LOG.error("THE OPTIMIZER WILL NOT BE RUN AT THIS TIME.")
            return {}

        forecasts = self.get_forecasts(now)
        parameters = self.get_parameters()

        results = self.optimizer(now, forecasts, parameters)
        if self.optimizer_debug_csv is not None:
            self.optimizer_debug_csv.writerow(now,
                                              results,
                                              forecasts,
                                              ["timestamp",
                                               "Optimization Status",
                                               "Objective Value",
                                               "Convergence Time"])

        commands = self.get_commands(results)
        if self.command_debug_csv is not None:
            self.command_debug_csv.writerow(now, commands)
        else:
            LOG.debug("Device commands: {}".format(commands))
        return commands
