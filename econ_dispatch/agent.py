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
from datetime import datetime, timedelta
from itertools import tee
import logging
import os

import pytz

from volttron.platform.agent import utils
from volttron.platform.vip.agent import Agent, Core
from volttron.platform.vip.agent.errors import Unreachable
from volttron.platform.jsonrpc import RemoteError
from volttron.platform.messaging import headers as headers_mod
from volttron.platform.scheduling import cron, periodic

from econ_dispatch.system_model import build_model_from_config
from econ_dispatch.forecast_models.history import Forecast as HistoryForecast
from econ_dispatch.utils import normalize_training_data

__version__ = '1.0.0'
__author1__ = 'Lee Burke <lee.burke@pnnl.gov>'
__author2__ = 'Kyle Monson <kyle.monson@pnnl.gov>'
__copyright__ = 'Copyright (c) 2018, Battelle Memorial Institute'
__license__ = 'Apache Version 2.0'

# utils.setup_logging()
logging.basicConfig(filename=os.path.expanduser('~/econ-dispatch/logs/econ_dispatch.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s: %(message)s')

LOG = logging.getLogger(__name__)


class EconDispatchAgent(Agent):
    """.. todo:: write agent description

    .. todo:: Define options -- maybe link to config man page

    :param kwargs: keyword arguments to Agent base class
    """
    def __init__(self,
                 optimization_schedule=60,
                 training_schedule=0,
                 schedule_start=None,
                 schedule_end=None,
                 make_reservations=False,
                 historian_vip_id="platform.historian",
                 simulation_mode=False,
                 offline_mode=False,
                 optimizer_debug=None,
                 command_debug=None,
                 optimizer_config={},
                 weather_config={},
                 forecast_configs={},
                 component_configs={},
                 **kwargs):
        super(EconDispatchAgent, self).__init__(**kwargs)

        self.make_reservations = make_reservations
        self.historian_vip_id = historian_vip_id
        self.simulation_mode = simulation_mode  # bool
        self.offline_mode = offline_mode  # may be bool or dict

        self.schedule_start = schedule_start
        self.schedule_end = schedule_end

        self.optimization_schedule = optimization_schedule
        self.next_optimization = None
        self.training_schedule = training_schedule
        self.historian_training = False
        self.next_training = None

        self.input_topics = set()
        self.remaining_simulation_inputs = set()
        self.all_topics = set()

        self.model = None
        self.optimization_greenlet = None
        self.training_greenlet = None
        self.default_config = dict(optimization_schedule=optimization_schedule,
                                   training_schedule=training_schedule,
                                   schedule_start=schedule_start,
                                   schedule_end=schedule_end,
                                   make_reservations=make_reservations,
                                   historian_vip_id=historian_vip_id,
                                   simulation_mode=simulation_mode,
                                   offline_mode=offline_mode,
                                   optimizer_debug=optimizer_debug,
                                   command_debug=command_debug,
                                   optimizer=optimizer_config,
                                   weather=weather_config,
                                   forecast_models=forecast_configs,
                                   component_models=component_configs)

        # Set a default configuration to ensure that self.configure is called
        # immediately to setup the agent.
        self.vip.config.set_default("config", self.default_config)

        # Hook self.configure up to changes to the configuration file "config".
        self.vip.config.subscribe(
            self.configure, actions=["NEW", "UPDATE"], pattern="config")

    def configure(self, config_name, action, contents):
        """Set agent parameters, subscribe to message bus, and schedule
        recurring events

        Called after the Agent has connected to the message bus. If a
        configuration exists at startup this will be called before onstart.

        Is called every time the configuration in the store changes.

        :param config_name: unused
        :param action: unused
        :param contents: contents of stored configuration
        """
        config = self.default_config.copy()
        config.update(contents)

        LOG.debug("Configuring Agent")

        # interactions with platform
        self.make_reservations = config.get("make_reservations", False)
        self.historian_vip_id = config.get(
            "historian_vip_id", "platform.historian")

        # debug files
        optimizer_debug = os.path.expanduser(config.get('optimizer_debug'))
        command_debug = os.path.expanduser(config.get('command_debug'))

        # required sections
        try:
            optimizer_config = config['optimizer']
            weather_config = config['weather']
            forecast_configs = config['forecast_models']
            component_configs = config['component_models']
        except KeyError as e:
            raise ValueError("Required section missing from config: {}"
                             "".format(repr(e)))

        # for co-simulation with EnergyPlus
        self.simulation_mode = bool(config.get("simulation_mode", False))

        # for fully independent simulation
        # allow offline_mode to be boolean or dict of params
        offline_mode = config.get("offline_mode", False)
        try:
            if isinstance(offline_mode, (bool, str)):
                self.offline_mode = bool(offline_mode)
                offline_config = {}
            elif isinstance(offline_mode, dict):
                offline_config = offline_mode
                self.offline_mode = bool(
                    offline_config.pop("offline_mode", True))
            else:
                raise ValueError
        except ValueError:
            LOG.error("problem reading offline_mode section of configuration")
            return

        if self.offline_mode and self.simulation_mode:
            LOG.error("simulation mode and offline mode are incompatible.")
            return

        # process offline_mode config
        if self.offline_mode:
            input_data = offline_config.get("input_data", None)

        # start and end should be UTC, timezone-naive (for `periodic`)
        start = config.get("schedule_start")
        if start is None:
            start = utils.get_aware_utc_now()
        else:
            start = utils.parse_timestamp_string(start)
            try:
                start = pytz.UTC.localize(start)
            except ValueError:
                pass
        # enforce timezone naive, UTC
        start = (start - start.utcoffset()).replace(tzinfo=None)
        self.schedule_start = start

        end = config.get("schedule_end")
        if end is not None:
            end = utils.parse_timestamp_string(end)
            try:
                end = pytz.UTC.localize(end)
            except ValueError:
                pass
            # enforce timezone naive, UTC (not if end is None)
            end = (end - end.utcoffset()).replace(tzinfo=None)
        self.schedule_end = end

        if end is None:
            LOG.debug("Running from {start}".format(start=start))
        else:
            LOG.debug("Running from {start} to {end}".format(start=start,
                                                             end=end))

        # schedules can be int (period in hours) or cron schedule
        training_schedule = config.get("training_schedule", 0)
        if training_schedule:
            self.historian_training = True
            try:
                # assume schedule is int, try `periodic`
                self.training_schedule = periodic(
                    timedelta(minutes=int(training_schedule)),
                    start=start,
                    stop=end)
            except ValueError:
                # assume `training_schedule` is valid cron string

                # cron does not always return start if start matches
                # warning: may return the minute before "start"
                if start is not None:
                    _start = start - timedelta(minutes=1)
                else:
                    _start = None
                self.training_schedule = cron(training_schedule,
                                              start=_start,
                                              end=end)
        else:
            # never train
            self.historian_training = False  # redundant, False by default
            self.training_schedule = iter(())

        optimization_schedule = config.get('optimization_schedule', 60)
        try:
            # assume `optimization_schedule` is int, try `periodic`
            self.optimization_schedule = periodic(
                timedelta(minutes=int(optimization_schedule)),
                start=start,
                stop=end)
        except ValueError:
            # assume `optimization_schedule` is valid cron string

            # cron does not always return start if start matches
            # warning: may return the minute before "start"
            if start is not None:
                _start = start - timedelta(minutes=1)
            else:
                _start = None
            self.optimization_schedule = cron(optimization_schedule,
                                              start=_start,
                                              stop=end)

        self.model = build_model_from_config(weather_config,
                                             optimizer_config,
                                             component_configs,
                                             forecast_configs,
                                             optimizer_debug,
                                             command_debug)

        if self.offline_mode:
            self._setup_timed_events()
            self.run_offline(input_data)
            # launch self.run_offline as greenlet
            # self.core.schedule(utils.get_aware_utc_now(),
            #                    self.run_offline,
            #                    input_data=input_data)
        else:
            # run online
            self.input_topics = self.model.get_component_input_topics()
            self.all_topics = self._create_all_topics(self.input_topics)
            self._create_subscriptions(self.all_topics)
            self.remaining_simulation_inputs = self.all_topics.copy()
            self._setup_timed_events()

    def _create_subscriptions(self, all_topics):
        """Subscribe to topics on message bus. If in simulation mode, subscribe
        to EnergyPlus topic

        :param all_topics: list of topics the agent will subscribe to
        """
        # Un-subscribe from everything.
        try:
            self.vip.pubsub.unsubscribe("pubsub", None, None)
        except KeyError:
            pass

        if all_topics:
            for topic in all_topics:
                # call `self._handle_publish` whenever a message is published
                # on a topic with prefix `topic`
                self.vip.pubsub.subscribe(peer='pubsub',
                                          prefix=topic,
                                          callback=self._handle_publish)

    def _create_all_topics(self, input_topics):
        """Convert list of topics in the form `base/point` to single
        `base/all` topic

        :param input_topics: list of input topics
        :rtype set:
        :returns: reduced set of `base/all` topics
        """
        all_topics = set()
        for topic in input_topics:
            base, _ = topic.rsplit('/', 1)
            all_topics.add(base+'/all')
        return all_topics

    def _handle_publish(
            self, peer, sender, bus, topic, headers, message):
        """Process messages posted to message bus

        :param peer: unused
        :param sender: unused
        :param bus: unused
        :param topic: topic of message in the form of `base/point`
        :param headers: message headers including timestamp
        :param message: body of message
        """
        base_topic, _ = topic.rsplit('/', 1)
        points = message[0]

        inputs = {}
        for point, value in points.items():
            point_topic = base_topic + '/' + point
            if point_topic in self.input_topics:
                inputs[point_topic] = value

        timestamp = utils.parse_timestamp_string(
            headers[headers_mod.TIMESTAMP])

        # assume unaware timestamps are UTC
        if (timestamp.tzinfo is None
                or timestamp.tzinfo.utcoffset(timestamp) is None):
            timestamp = pytz.utc.localize(timestamp)

        if inputs:
            self.model.process_inputs(timestamp, inputs)

        if self.simulation_mode:
            if self.historian_training:
                self.train_components(timestamp)

            if topic in self.remaining_simulation_inputs:
                self.remaining_simulation_inputs.remove(topic)
            else:
                LOG.warning("Duplicate inputs: {}".format(topic))

            if not self.remaining_simulation_inputs:
                LOG.info("Run triggered by all input topics receiving publish")
                # if not enough time has passed, all input topics will need
                # to be received *again*
                self.remaining_simulation_inputs = self.all_topics.copy()
                self.run_optimizer(timestamp)

    def _setup_timed_events(self):
        """Schedule recurring events using Volttron vip.core.schedule"""
        # Don't setup the greenlets if we are offline/driven by simulation.
        if not self.simulation_mode and not self.offline_mode:
            # initialize self.next_optimization
            # be careful not to mutate self.*_schedule
            self.optimization_schedule, optimization_schedule_clone = \
                tee(self.optimization_schedule)
            try:
                self.next_optimization = pytz.UTC.localize(
                    next(optimization_schedule_clone))
                LOG.info("Next optimization scheduled for {}"
                         "".format(self.next_optimization))
            except StopIteration:
                self.next_optimization = None
                LOG.error("No optimizations scheduled")
            # initialize self.next_training
            # be careful not to mutate self.*_schedule
            self.training_schedule, training_schedule_clone = \
                tee(self.training_schedule)
            try:
                self.next_training = pytz.UTC.localize(
                    next(training_schedule_clone))
                LOG.info("Next training scheduled for {}"
                         "".format(self.next_training))
            except StopIteration:
                self.next_training = None
                LOG.info("No trainings scheduled")

            if self.optimization_greenlet is not None:
                self.optimization_greenlet.kill()
                self.optimization_greenlet = None
            self.optimization_greenlet = self.core.schedule(
                self.optimization_schedule, self.run_optimizer)

            if self.training_greenlet is not None:
                self.training_greenlet.kill()
                self.training_greenlet = None
            self.training_greenlet = self.core.schedule(
                self.training_schedule, self.train_components)
        else:
            # initialize self.next_optimization
            try:
                self.next_optimization = pytz.UTC.localize(
                    next(self.optimization_schedule))
                LOG.info("Next optimization scheduled for {}"
                         "".format(self.next_optimization))
            except StopIteration:
                self.next_optimization = None
                LOG.error("No optimizations scheduled")
            # initialize self.next_training
            try:
                self.next_training = pytz.UTC.localize(
                    next(self.training_schedule))
                LOG.info("Next training scheduled for {}"
                         "".format(self.next_training))
            except StopIteration:
                self.next_training = None
                LOG.info("No trainings scheduled")

    def run_offline(self, input_data=None):
        """TODO: write docstring for offline mode

        :param input_data:
        """
        if input_data is not None:
            LOG.debug("Processing input data file {}".format(input_data))
            try:
                inputs = HistoryForecast()
                training_data = normalize_training_data(input_data)
                inputs.train(training_data)
            except Exception as e:
                LOG.error(repr(e))
                return

        while self.next_optimization is not None:
            now = self.next_optimization
            LOG.debug("Processing timestamp: " + str(now))

            if input_data is not None:
                current_inputs = inputs.derive_variables(now)
                self.model.process_inputs(now, current_inputs)

            if self.next_training is not None and now >= self.next_training:
                LOG.debug("Training components")
                self.train_components()

            LOG.debug("Running hourly optimizer")
            self.run_optimizer(now)

        LOG.info("Offline mode complete")

    def run_optimizer(self, now=None):
        """Run the optimizer and set commands"""
        if now is None:
            now = utils.get_aware_utc_now()

        if (self.next_optimization is not None
                and self.next_optimization > now):
            LOG.debug("Not running optimizer: not enough time has passed")
            return

        LOG.info("Running optimization for {}".format(now))
        try:
            self.next_optimization = pytz.UTC.localize(
                next(self.optimization_schedule))
            LOG.info("Next optimization scheduled for {}"
                     "".format(self.next_optimization))
        except StopIteration:
            self.next_optimization = None
            LOG.info("No more optimizations scheduled")

        commands = self.model.run_optimizer(now)

        if commands:
            if not self.offline_mode:
                if self.make_reservations:
                    self.reserve_actuator(commands)
                    self.actuator_set(commands)
                    self.reserve_actuator_cancel()
                else:
                    self.actuator_set(commands)

    def train_components(self, now=None):
        """Gather training parameters, query historian for training data,
        then pass data to model for further processing

        :param now: timestamp for simulation mode
        """
        if self.next_training is None:
            return

        if now is None:
            # We are being driven by a greenlet in realtime
            # Always run when we are told
            now = utils.get_aware_utc_now()
        else:
            # We are being driven by simulation
            # Return if we are not ready to run again
            if self.next_training > now:
                return

        # Train both forecast and component models
        for forecast_models in (False, True):
            results = {}
            all_parameters = self.model.get_training_parameters(forecast_models)

            for name, parameters in all_parameters.items():
                window, sources = parameters
                end = now
                start = end - timedelta(days=window)
                training_data = {}
                for topic in sources:
                    value = self.vip.rpc.call(self.historian_vip_id,
                                              "query",
                                              topic,
                                              utils.format_timestamp(start),
                                              utils.format_timestamp(end),
                                             ).get(timeout=4)
                    training_data[topic] = value

                results[name] = training_data

            self.model.apply_all_training_data(results, forecast_models)

        try:
            self.next_training = pytz.UTC.localize(
                next(training_schedule))
            LOG.info("Next training scheduled for {}"
                     "".format(self.next_training))
        except StopIteration:
            self.next_training = None
            self.historian_training = False
            LOG.info("No more trainings scheduled")

    def reserve_actuator(self, topic_values):
        """Call the actuator's request_new_schedule method to get device
        schedule

        :param topic_values: list of topics like `base/point`, where base is
            an actuation device
        :returns: Return result from request_new_schedule method
            and True or False for error in scheduling device.
        :rtype: bool
        :raises: LockError

        .. warning:: Calling without previously scheduling a device and not
            within the time allotted will raise a LockError
        """
        success = True

        if not self.make_reservations:
            return success

        # TODO: can we use the usual utils.format_timestamp for this?
        DATE_FORMAT = '%m-%d-%y %H:%M'

        _now = datetime.now()
        str_now = _now.strftime(DATE_FORMAT)
        _end = _now + timedelta(minutes=1)
        str_end = _end.strftime(DATE_FORMAT)

        # build list of unique schedule requests from topic list
        schedule_request = set()
        for topic in topic_values:
            actuation_device, _ = topic.rsplit('/', 1)
            schedule_request.add((actuation_device, str_now, str_end))
        schedule_request = list(schedule_request)

        try:
            result = self.vip.rpc.call('platform.actuator',
                                       'request_new_schedule',
                                       "",
                                       "econ_dispatch",
                                       'HIGH',
                                       schedule_request).get(timeout=4)
        except RemoteError as ex:
            LOG.warning("Failed to create actuator schedule (RemoteError): "
                        "{}".format(str(ex)))
            return False
        except Unreachable:
            LOG.error("Unable to reach platform.actuator")
            return False

        if result['result'] == 'FAILURE':
            if result['info'] == 'TASK_ID_ALREADY_EXISTS':
                LOG.info('Task to schedule device already exists')
                success = True
            else:
                LOG.warn('Failed to schedule devices (unavailable)')
                success = False

        return success

    def reserve_actuator_cancel(self):
        """Cancel actuator reservations"""
        if self.make_reservations:
            try:
                self.vip.rpc.call('platform.actuator',
                                  'request_cancel_schedule',
                                  "",
                                  "econ_dispatch").get(timeout=4)
            except RemoteError as ex:
                LOG.warning("Failed to cancel schedule (RemoteError): "
                            "{}".format(str(ex)))
            except Unreachable:
                LOG.error("Unable to reach platform.actuator")

    def actuator_set(self, topic_values):
        """Call the actuator's set_point method to set point on device

        :param topic_values: Key value pairs of what is to be written
        """

        try:
            result = self.vip.rpc.call('platform.actuator',
                                       'set_multiple_points',
                                       "",
                                       list(topic_values.items())).get(timeout=4)
        except RemoteError as ex:
            LOG.error("Error occured in actuator set: {}. Failed to set: {}"
                      "".format(str(ex), topic_values))
        except Unreachable:
            LOG.error("Unable to reach platform.actuator. Failed to set: {}"
                      "".format(topic_values))

def econ_dispatch_agent(config_path, **kwargs):
    """Parses agent configuration to run driven agent.

    :param config_path: path to Volttron-standard config file
        (json with python-style comments)
    :param kwargs: Keyword arguments for Agent base class"""
    config = utils.load_config(config_path)

    # schedules can be int (period in hours), cron schedule, or null
    optimization_schedule = config.get('optimization_schedule', 60)
    training_schedule = config.get("training_schedule", 0)

    schedule_start = config.get("schedule_start")
    schedule_end = config.get("schedule_end")

    # interactions with platform
    make_reservations = config.get("make_reservations", False)
    historian_vip_id = config.get("historian_vip_id", "platform.historian")
    # for co-simulation with EnergyPlus
    simulation_mode = bool(config.get("simulation_mode", False))
    # for fully independent simulation
    offline_mode = config.get("offline_mode", False)

    # debug files
    optimizer_debug = config.get('optimizer_debug')
    command_debug = config.get("command_debug")

    # required sections
    try:
        optimizer_config = config['optimizer']
        weather_config = config['weather']
        forecast_configs = config['forecast_models']
        component_configs = config['component_models']
    except KeyError as e:
        raise ValueError("Required section missing from config: {}"
                            "".format(repr(e)))

    LOG.debug("Launching agent")

    return EconDispatchAgent(optimization_schedule=optimization_schedule,
                             training_schedule=training_schedule,
                             schedule_start=schedule_start,
                             schedule_end=schedule_end,
                             make_reservations=make_reservations,
                             historian_vip_id=historian_vip_id,
                             simulation_mode=simulation_mode,
                             offline_mode=offline_mode,
                             optimizer_debug=optimizer_debug,
                             command_debug=command_debug,
                             optimizer_config=optimizer_config,
                             weather_config=weather_config,
                             forecast_configs=forecast_configs,
                             component_configs=component_configs,
                             **kwargs)

def main():
    """Launch agent using Volttron VIP wrapper"""
    utils.vip_main(econ_dispatch_agent)

if __name__ == '__main__':
    # Entry point for script
    main()
