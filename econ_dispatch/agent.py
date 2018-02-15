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

from application import Application

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
    arguments = config.get('arguments')
    mode = True if config.get('mode', 'PASSIVE') == 'ACTIVE' else False
    campus_building_config = config['device']
    analysis_name = campus_building_config.get('analysis_name', 'analysis_name')
    analysis_dict = {'analysis_name': analysis_name}
    arguments.update(analysis_dict)
    campus_building = dict((key, campus_building_config[key]) for key in ['campus', 'building'])
    analysis = deepcopy(campus_building)
    analysis.update(analysis_dict)
    device_config = campus_building_config['unit']
    #command_devices = device_config[:]
    device_topic_dict = {}
    device_topic_list = []
    for device_name in device_config:
        device_topic = topics.DEVICES_VALUE(campus=campus_building.get('campus'),
                                            building=campus_building.get('building'),
                                            unit=device_name,
                                            path='',
                                            point='all')
        device_topic_dict.update({device_topic: device_name})
        device_topic_list.append(device_name)

    base_actuator_path = topics.RPC_DEVICE_PATH(campus=campus_building.get('campus', ''),
                                               building=campus_building.get('building', ''),
                                               unit=None,
                                               path='',
                                               point=None)

    output_file_prefix = config.get('output_file_prefix')

    command_output_file = config.get('command_output_file')

    if command_output_file is not None:
        command_output_file = open(command_output_file, "wb")

    make_reservations = config.get("make_reservations", False)

    # This instances is used to call the applications run method when
    # data comes in on the message bus.  It is constructed here
    # so that_process_results each time run is called the application
    # can keep it state.
    app_instance = Application(**arguments)

    
    class DrivenAgent(Agent):
        """Agent listens to message bus device and runs when data is published.
        """
        
        def __init__(self, **kwargs):
            """
            Initializes agent
            :param kwargs: Any driver specific parameters"""
        
            super(DrivenAgent, self).__init__(**kwargs)

            # master is where we copy from to get a poppable list of
            # subdevices that should be present before we run the analysis.
            self._master_devices = device_topic_list
            self._needed_devices = []
            self._device_values = {}
            self._initialize_devices()
            self.received_input_datetime = None
            self._kwargs = kwargs
            self._command_output_csv = None

            self._header_written = False
            self.file_creation_set = set()

        def _initialize_devices(self):
            self._needed_devices = deepcopy(self._master_devices)
            self._device_values = {}

        @Core.receiver('onstart')
        def starup(self, sender, **kwargs):
            """
            Starts up the agent and subscribes to device topics 
            based on agent configuration.
            :param sender: 
            :param kwargs: Any driver specific parameters
            :type sender: str"""
            self._initialize_devices()
            for device_topic in device_topic_dict:
                _log.debug('Subscribing to ' + device_topic)
                self.vip.pubsub.subscribe(peer='pubsub',
                                          prefix=device_topic,
                                          callback=self.on_analysis_message)
        
        def _should_run_now(self):
            """
            Checks if messages from all the devices are received 
                before running application         
            :returns: True or False based on received messages.
            :rtype: boolean"""
            # Assumes the unit/all values will have values.
            if not len(self._device_values.keys()) > 0:
                return False
            return not len(self._needed_devices) > 0
            
        def on_analysis_message(self, peer, sender, bus, topic, headers, message):
            """
            Subscribe to device data and assemble data set to pass
                to applications.
            :param peer: 
            :param sender: device name
            :param bus: 
            :param topic: device path topic 
            :param headers: message headers
            :param message: message containing points and values dict 
                    from device with point type
            :type peer: str
            :type sender: str
            :type bus: str
            :type topic: str
            :type headers: dict
            :type message: dict"""
            
            device_data = message[0]
            if isinstance(device_data, list):
                device_data = device_data[0]

            def aggregate_subdevice(device_data):
                tagged_device_data = {}
                device_tag = device_topic_dict[topic]
                if device_tag not in self._needed_devices:
                    return False
                for key, value in device_data.items():
                    device_data_tag = '&'.join([key, device_tag])
                    tagged_device_data[device_data_tag] = value
                self._device_values.update(tagged_device_data)
                self._needed_devices.remove(device_tag)
                return True

            device_needed = aggregate_subdevice(device_data)
            if not device_needed:
                _log.error("Warning device values already present, "
                           "reinitializing")

            if self._should_run_now():
                field_names = {}
                for k, v in self._device_values.items():
                    field_names[k.lower() if isinstance(k, str) else k] = v

                _timestamp = utils.parse_timestamp_string(headers[headers_mod.TIMESTAMP])
                self.received_input_datetime = _timestamp

                device_data = field_names
                results = app_instance.run(_timestamp, device_data)
                # results = app_instance.run(
                # dateutil.parser.parse(self._subdevice_values['Timestamp'],
                #                       fuzzy=True), self._subdevice_values)
                self._process_results(_timestamp, results)
                self._initialize_devices()
            else:
                _log.info("Still need {} before running.".format(self._needed_devices))

        def _process_results(self, timestamp, results):
            """
            Runs driven application with converted data. Calls appropriate 
                methods to process commands, log and table_data in results.
            :param results: Results object containing commands for devices, 
                    log messages and table data.
            :type results: Results object \\volttron.platform.agent.driven
            :returns: Same as results param. 
            :rtype: Results object \\volttron.platform.agent.driven"""

            topic_value = self.create_topic_values(results)

            _log.debug('Processing Results!')
            if mode:
                _log.debug("ACTUATE ON DEVICE.")
                actuator_error = False
                if make_reservations and results.devices:
                    results, actuator_error = self.actuator_request(results)
                if not actuator_error:
                    self.actuator_set(topic_value)
                if make_reservations and results.devices and not actuator_error:
                    self.actuator_cancel()

            for value in results.log_messages:
                _log.debug("LOG: {}".format(value))
            for key, value in results.table_output.items():
                _log.debug("TABLE: {}->{}".format(key, value))
            if output_file_prefix is not None:
                results = self.create_file_output(results)
            if command_output_file is not None:
                self.create_command_file_output(timestamp, topic_value)
            # if len(results.table_output.keys()):
            #     results = self.publish_analysis_results(results)
            return results
        
        # def publish_analysis_results(self, results):
        #     """
        #     Publish table_data in analysis results to the message bus for
        #         capture by the data historian.
        #
        #     :param results: Results object containing commands for devices,
        #             log messages and table data.
        #     :type results: Results object \\volttron.platform.agent.driven
        #     :returns: Same as results param.
        #     :rtype: Results object \\volttron.platform.agent.driven"""
        #
        #     headers = {
        #         headers_mod.CONTENT_TYPE: headers_mod.CONTENT_TYPE.JSON,
        #         headers_mod.DATE: str(self.received_input_datetime),
        #     }
        #     for app, analysis_table in results.table_output.items():
        #         try:
        #             name_timestamp = app.split('&')
        #             _name = name_timestamp[0]
        #             timestamp = name_timestamp[1]
        #         except:
        #             _name = app
        #             timestamp = str(self.received_input_datetime)
        #         headers = {
        #             headers_mod.CONTENT_TYPE: headers_mod.CONTENT_TYPE.JSON,
        #             headers_mod.DATE: timestamp,
        #         }
        #         for entry in analysis_table:
        #             for key, value in entry.items():
        #                 for _device in command_devices:
        #                     analysis['unit'] = _device
        #                     analysis_topic = topics.ANALYSIS_VALUE(point=key, **analysis)
        #                     datatype = 'float'
        #                     if isinstance(value, int):
        #                         datatype = 'int'
        #                     kbase = key[key.rfind('/') + 1:]
        #                     message = [{kbase: value},
        #                                {kbase: {'tz': 'US/Pacific',
        #                                         'type': datatype,
        #                                         'units': 'float',
        #                                         }
        #                                 }]
        #                     self.vip.pubsub.publish(
        #                         'pubsub', analysis_topic, headers, message)
        #     return results

        def create_command_file_output(self, timestamp, topic_value):
            if not topic_value:
                return

            if self._command_output_csv is None:
                field_names = ["timestamp"] + list(topic_value.keys())
                self._command_output_csv = csv.DictWriter(command_output_file, field_names)
                self._command_output_csv.writeheader()

            tv_copy = topic_value.copy()
            tv_copy["timestamp"] = utils.format_timestamp(timestamp)
            self._command_output_csv.writerow(tv_copy)
        
        def create_file_output(self, results):
            """
            Create results/data files for testing and algorithm validation 
            if table data is present in the results.
            
            :param results: Results object containing commands for devices, 
                    log messages and table data.
            :type results: Results object \\volttron.platform.agent.driven
            :returns: Same as results param. 
            :rtype: Results object \\volttron.platform.agent.driven"""
            for key, value in results.table_output.items():
                name_timestamp = key.split('&')
                _name = name_timestamp[0]
                timestamp = name_timestamp[1]
                file_name = output_file_prefix + "-" + _name + ".csv"
                if file_name not in self.file_creation_set:
                    self._header_written = False
                self.file_creation_set.update([file_name])
                for row in value:
                    with open(file_name, 'a+') as file_to_write:
                        row.update({'Timestamp': timestamp})
                        _keys = row.keys()
                        file_output = csv.DictWriter(file_to_write, _keys)
                        if not self._header_written:
                            file_output.writeheader()
                            self._header_written = True
                        file_output.writerow(row)
                    file_to_write.close()
            return results
        
        def actuator_request(self, results):
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
            
            _now = dt.now()
            str_now = _now.strftime(DATE_FORMAT)
            _end = _now + td(minutes=1)
            str_end = _end.strftime(DATE_FORMAT)
            schedule_request = []
            for _device in results.devices:
                actuation_device = base_actuator_path(unit=_device, point='')
                schedule_request.append([actuation_device, str_now, str_end])

            try:
                result = self.vip.rpc.call('platform.actuator',
                                           'request_new_schedule',
                                           "", "driven.agent.write", 'HIGH',
                                           schedule_request).get(timeout=4)
            except RemoteError as ex:
                _log.warning("Failed to schedule device {} (RemoteError): {}".format(_device, str(ex)))
                request_error = True

            if result['result'] == 'FAILURE':
                if result['info'] =='TASK_ID_ALREADY_EXISTS':
                    _log.info('Task to schedule device already exists ' + _device)
                    request_error = False
                else:
                    _log.warn('Failed to schedule device (unavailable) ' + _device)
                    request_error = True
            else:
                request_error = False

            return results, request_error

        def actuator_cancel(self):
            self.vip.rpc.call('platform.actuator',
                               'request_cancel_schedule',
                               "", "driven.agent.write").get(timeout=4)
        
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

        def create_topic_values(self, results):
            topic_values = {}
            for device, point_value_dict in results.devices.items():
                for point, new_value in point_value_dict.items():
                    point_path = base_actuator_path(unit=device, point=point)
                    topic_values[str(point_path)] = new_value
            return topic_values

    DrivenAgent.__name__ = 'DrivenLoggerAgent'
    return DrivenAgent(**kwargs)


def main(argv=sys.argv):
    ''' Main method.'''
    utils.vip_main(econ_dispatch_agent)


if __name__ == '__main__':
    # Entry point for script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass