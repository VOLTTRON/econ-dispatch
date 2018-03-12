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
import csv
from datetime import datetime as dt, timedelta as td
import logging
from copy import deepcopy
import sys

from volttron.platform.agent import utils
from volttron.platform.vip.agent import Agent, Core
from volttron.platform.jsonrpc import RemoteError
from volttron.platform.messaging import (headers as headers_mod, topics)

from system_model import build_model_from_config

__version__ = '3.0.0'
__author4__ = 'Kyle Monson <kyle.monson@pnnl.gov>'
__copyright__ = 'Copyright (c) 2018, Battelle Memorial Institute'
__license__ = 'Apache Version 2.0'
DATE_FORMAT = '%m-%d-%y %H:%M'

utils.setup_logging()
_log = logging.getLogger(__name__)

def econ_dispatch_agent(config_path, **kwargs):
    """Reads agent configuration and converts it to run driven agent.
    :param kwargs: Any driver specific parameters"""
    config = utils.load_config(config_path)

    training_frequency = int(config.get("training_frequency", 7))
    make_reservations = config.get("make_reservations", False)
    optimizer_debug = config.get('optimizer_debug')
    optimization_frequency = int(config.get('optimization_frequency', 60))
    optimizer = config.get('optimizer', {})
    weather = config.get("weather", {})
    forecast_models = config.get("forecast_models", {})
    components = config.get("components", {})

    return EconDispatchAgent(make_reservations=make_reservations,
                             optimization_frequency=optimization_frequency,
                             optimizer=optimizer,
                             optimizer_debug=optimizer_debug,
                             weather=weather,
                             forecast_models=forecast_models,
                             components=components,
                             training_frequency=training_frequency,
                             **kwargs)

    
class EconDispatchAgent(Agent):
    """Agent listens to message bus device and runs when data is published.
    """

    def __init__(self,
                 make_reservations=False,
                 optimization_frequency=60,
                 optimizer={},
                 optimizer_debug=None,
                 weather={},
                 forecast_models={},
                 components={},
                 training_frequency=7,
                 **kwargs):
        """
        Initializes agent
        :param kwargs: Any driver specific parameters"""

        super(EconDispatchAgent, self).__init__(**kwargs)

        self.model = None
        self.make_reservations = make_reservations
        self.training_frequency = training_frequency
        self.input_topics = set()
        self.inputs = {}

        # master is where we copy from to get a poppable list of
        # subdevices that should be present before we run the analysis.
        self.default_config = dict(make_reservations=make_reservations,
                                 optimization_frequency=optimization_frequency,
                                 optimizer=optimizer,
                                 optimizer_debug=optimizer_debug,
                                 weather=weather,
                                 forecast_models=forecast_models,
                                 components=components,
                                 training_frequency=training_frequency)

        # Set a default configuration to ensure that self.configure is called immediately to setup
        # the agent.
        self.vip.config.set_default("config", self.default_config)
        # Hook self.configure up to changes to the configuration file "config".
        self.vip.config.subscribe(self.configure, actions=["NEW", "UPDATE"], pattern="config")


    def configure(self, config_name, action, contents):
        """
        Called after the Agent has connected to the message bus. If a configuration exists at startup
        this will be called before onstart.

        Is called every time the configuration in the store changes.
        """
        config = self.default_config.copy()
        config.update(contents)

        _log.debug("Configuring Agent")

        try:
            make_reservations = bool(config.get("make_reservations", False))
            weather_config = config["weather"],
            optimizer_config = config["optimizer"]
            component_configs = config["components"]
            forecast_model_configs = config["forecast_models"]
            optimization_frequency = int(config.get("optimization_frequency", 60))
            optimizer_csv_filename = config.get("optimizer_debug")
            training_frequency = int(config.get("training_frequency", 7))
        except (ValueError, KeyError) as e:
            _log.error("ERROR PROCESSING CONFIGURATION: {}".format(e))
            return

        self.make_reservations = make_reservations
        self.training_frequency = training_frequency

        self.model = build_model_from_config(weather_config,
                                             optimizer_config,
                                             component_configs,
                                             forecast_model_configs,
                                             optimization_frequency,
                                             optimizer_csv_filename)

        self.input_topics = self.model.get_component_input_topics()
        self.all_topics = self._create_all_topics(self.input_topics)
        self._create_subscriptions(self.all_topics)

    def _create_subscriptions(self, all_topics):
        # Un-subscribe from everything.
        try:
            self.vip.pubsub.unsubscribe("pubsub", None, None)
        except KeyError:
            pass

        for topic in all_topics:
            self.vip.pubsub.subscribe(peer='pubsub',
                                      prefix=topic,
                                      callback=self._handle_publish)

    def _create_all_topics(self, input_topics):
        all_topics = set()
        for topic in input_topics:
            base, point = topic.rsplit('/', 1)
            all_topics.add(base+'/all')
        return all_topics

    def _handle_publish(self, peer, sender, bus, topic, headers,
                        message):
        base_topic, _ = topic.rsplit('/', 1)
        points = message[0]

        for point, value in points.iteritems():
            topic = base_topic + '/' + point
            if topic in self.input_topics:
                self.inputs[topic] = value

        if self.inputs:
            timestamp = utils.parse_timestamp_string(headers[headers_mod.TIMESTAMP])
            commands = self.model.run(timestamp, self.inputs)

            if commands and self.reserve_actuator(commands):
                self.actuator_set(commands)
                self.reserve_actuator_cancel()


    def reserve_actuator(self, topic_values):
        """
        Calls the actuator's request_new_schedule method to get
                device schedule

        :param results: Results object containing commands for devices,
                log messages and table data.
        :type results: Results object \\volttron.platform.agent.driven
        :returns: Return result from request_new_schedule method
                    and True or False for error in scheduling device.
        :rtype: dict and boolean
        :Return Values:

        The return values has the following format:

            result = {'info': u'', 'data': {}, 'result': 'SUCCESS'}
            request_error = True/False

        warning:: Calling without previously scheduling a device and not within
                     the time allotted will raise a LockError"""

        success = True

        if not self.make_reservations:
            return success

        _now = dt.now()
        str_now = _now.strftime(DATE_FORMAT)
        _end = _now + td(minutes=1)
        str_end = _end.strftime(DATE_FORMAT)
        schedule_request = set()
        for topic in topic_values:
            actuation_device, _ = topic.rsplit('/', 1)
            schedule_request.add((actuation_device, str_now, str_end))

        schedule_request = list(schedule_request)

        try:
            result = self.vip.rpc.call('platform.actuator',
                                       'request_new_schedule',
                                       "", "econ_dispatch", 'HIGH',
                                       schedule_request).get(timeout=4)
        except RemoteError as ex:
            _log.warning("Failed to create actuator schedule (RemoteError): {}".format(str(ex)))
            return False

        if result['result'] == 'FAILURE':
            if result['info'] =='TASK_ID_ALREADY_EXISTS':
                _log.info('Task to schedule device already exists')
                success = True
            else:
                _log.warn('Failed to schedule devices (unavailable)')
                success = False

        return success

    def reserve_actuator_cancel(self):
        if self.make_reservations:
            try:
                self.vip.rpc.call('platform.actuator',
                                   'request_cancel_schedule',
                                   "", "econ_dispatch").get(timeout=4)
            except RemoteError as ex:
                _log.warning("Failed to cancel schedule (RemoteError): {}".format(str(ex)))

    def actuator_set(self, topic_values):
        """
        Calls the actuator's set_point method to set point on device

        :param topic_values: Key value pairs of what is to be written."""

        try:
            result = self.vip.rpc.call('platform.actuator', 'set_multiple_points',
                                       "", topic_values.items()).get(timeout=4)
        except RemoteError as ex:
            _log.warning("Failed to set: {}".format(str(ex)))

        for topic, ex in result.iteritems():
            _log.warning("Failed to set {}: {}".format(topic, str(ex)))






def main(argv=sys.argv):
    ''' Main method.'''
    utils.vip_main(econ_dispatch_agent)


if __name__ == '__main__':
    # Entry point for script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass