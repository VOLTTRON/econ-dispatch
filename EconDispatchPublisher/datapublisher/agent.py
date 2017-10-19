# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2016, Battelle Memorial Institute
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
import datetime
from dateutil import parser
import logging
import os
import re
import sys

from volttron.platform.vip.agent import *
from volttron.platform.agent import utils
from volttron.platform.agent.utils import jsonapi
from volttron.platform.messaging import topics
from volttron.platform.messaging import headers as headers_mod

_log = logging.getLogger(__name__)
__version__ = '4.0.0'

HEADER_NAME_DATE = headers_mod.DATE
HEADER_NAME_CONTENT_TYPE = headers_mod.CONTENT_TYPE
VALUE_RESPONSE_PREFIX = topics.ACTUATOR_VALUE()
ERROR_RESPONSE_PREFIX = topics.ACTUATOR_ERROR()
SCHEDULE_RESPONSE_SUCCESS = 'SUCCESS'
SCHEDULE_RESPONSE_FAILURE = 'FAILURE'
SCHEDULE_ACTION_NEW = 'NEW_SCHEDULE'
SCHEDULE_ACTION_CANCEL = 'CANCEL_SCHEDULE'

__authors__ = ['Robert Lutes <robert.lutes@pnnl.gov>',
               'Kyle Monson <kyle.monson@pnnl.gov>',
               'Craig Allwardt <craig.allwardt@pnnl.gov>']
__copyright__ = 'Copyright (c) 2016, Battelle Memorial Institute'
__license__ = 'FreeBSD'


def DataPub(config_path, **kwargs):
    '''Emulate device driver to publish data and Actuatoragent for testing.

    The first column in the data file must be the timestamp and it is not
    published to the bus unless the config option:
    'use_timestamp' - True will use timestamp in input file.
    timestamps. False will use the current now time and publish using it.
    '''
    conf = utils.load_config(config_path)
    use_timestamp = conf.get('use_timestamp', True)
    remember_playback = conf.get('remember_playback', False)
    reset_playback = conf.get('reset_playback', False)

    custom_topic = conf.get('custom_topic')
    pub_interval = float(conf.get('publish_interval', 5))
    if not custom_topic:
        device_path = (
            ''.join([conf.get('campus'), '/', conf.get('building'), '/']))

        BASETOPIC = conf.get('basetopic')
        # device root is the root of the publishing tree.
        device_root = ''.join([BASETOPIC, '/', device_path])
    else:
        device_root = custom_topic

    path = conf.get('input_file')
    if not os.path.exists(path):
        raise ValueError('Invalid input file specified.')

    # if unit is a string then there aren't any subdevices and we
    # just use the name of the device as is.
    unit = conf.get('unit')
    
    # unittype_map maps the point name to the proper units.
    unittype_map = conf.get('unittype_map', {})
    
    # should we keep playing the file over and over again.
    replay_data = conf.get('replay_data', False)
    
    header_point_map = {}
    # If thie unit is a dictionary then the device
    if isinstance(unit, dict):
        # header point map maps the prefix of a column in the csv file to
        # the relative container.  For example if the csv file column in
        # question is FCU13259_Heating and FCU13259 is under an rtu5 then
        # the key FCU13259 would map to rtu5/FCU13259.  This will make it
        # trivial later to append the sensor_name Heating to the relative
        # point to publish the value.
        for prefix, v in unit.items():
            # will allow publishing under root level items such
            # as rtu5_Compensator to rtu5/Compensator
            header_point_map[prefix] = prefix

            for prefix2 in v['subdevices']:
                # will allow publishing to subdevices such as
                # FCU13259_HeatingSignal to rtu5/FCU13259/HeatingSignal
                header_point_map[prefix2] = '/'.join([prefix, prefix2])

    class Publisher(Agent):
        '''Simulate real device.  Publish csv data to message bus.

        Configuration consists of csv file and publish topic
        '''
        def __init__(self, **kwargs):
            '''Initialize data publisher class attributes.'''
            super(Publisher, self).__init__(**kwargs)
            self._src_file_handle = open(path, 'rb')

            # Uses dictreader so that thee first line in the file is auto
            # ingested and becaums the headers for the dictionary.  Use the
            # fieldnames property to get the names of the fields available.
            self._reader = csv.DictReader(self._src_file_handle,
                                          delimiter=',')

            self.end_time = None
            self.start_time = None
            self.task_id = None
            utils.setup_logging()
            self._log = logging.getLogger(__name__)
            self.scheduled_event = None
            logging.basicConfig(
                level=logging.debug,
                format='%(asctime)s   %(levelname)-8s %(message)s',
                datefmt='%m-%d-%y %H:%M:%S')
            if remember_playback:
                self._log.info('Keeping track of line being played in case of interuption.')
            else:
                self._log.info('Not storing line being played (enable by setting remember_playback=1 in config file')
            self._log.info('Publishing Starting')
            self._line_on = 0
            start_line = self.get_start_line()
            
            # Only move the start_line if the reset_playback switch is off and
            # the remember_playback switch is on.
            if not reset_playback and remember_playback:                        
                while self._line_on - 1 < start_line:
                    self._reader.next()
                    self._line_on+=1
                        
            self._log.info('Playback starting on line: {}'.format(self._line_on))
                    
                
            
        def store_line_on(self):
            basename = os.path.basename(path)+'.count'
            with open(basename, 'wb') as fd:
                fd.write(str(self._line_on))
                fd.close()
        
        def get_start_line(self):
            basename = os.path.basename(path)+'.count'
            try:
                with open(basename, 'rb') as fd:
                    count = fd.read()
                    fd.close()
                
                return int(count)
            except Exception as e:
                print(e.message)
                
                return 0
            
        def remove_store_line(self):
            basename = os.path.basename(path)+'.count'
            if os.path.exists(basename):
                try:
                    os.remove(basename)
                except:
                    self._log.info('Unable to remove line store.')

        @staticmethod
        def _get_unit(point):
            ''' Get a unit type based upon the regular expression in the config file.

                if NOT found returns percent as a default unit.
            '''
            for k, v in unittype_map.items():
                if re.match(k, point):
                    return v
            return 'percent'

        @staticmethod
        def _publish_point(topic, point, data):
            # makesure topic+point gives a true value.
            if not topic.endswith('/') and not point.startswith('/'):
                topic += '/'

            # Transform the values into floats rather than the read strings.
            if not isinstance(data, dict):
                data = {point: float(data)}

            # Create metadata with the type, tz ... in it.
            meta = {}
            topic_point = topic + point
            for p, v in data.items():
                meta[p] = {'type': 'float', 'tz': 'US/Pacific', 'units': self._get_unit(p)}

            # Message will always be a list of two elements.  The first element
            # is set with the data to be published.  The second element is meta
            # data.  The meta data must hold a type element in order for the
            # historians to work properly.
            message = [data, meta]

            self.vip.pubsub.publish(peer='pubsub',
                                    topic=topic_point,
                                    message=message,  # [data, {'source': 'publisher3'}],
                                    headers=headers).get(timeout=2)

                    
        @Core.receiver('onstart')
        def onstart(self):
            '''Publish data from file to message bus.'''
            data = {}


            if self._src_file_handle is not None \
                    and not self._src_file_handle.closed:
                
                try:                    
                    data = self._reader.next()
                    self._line_on+=1
                    if remember_playback:
                        self.store_line_on()
                except StopIteration:
                    if replay_data:
                        _log.info('Restarting player at the begining of the file.')
                        self._src_file_handle.seek(0)
                        self._line_on = 0
                        if remember_playback:
                            self.store_line_on()
                    else:                        
                        self._src_file_handle.close()
                        self._src_file_handle = None
                        _log.info("Completed publishing all records for file!")
                        self.core.stop()
                                            
                        return
                # break out if no data is left to be found.
                if not data:
                    self._src_file_handle.close()
                    self._src_file_handle = None
                    return

                if use_timestamp:
                    headers = {HEADER_NAME_DATE: data['Timestamp']}
                else:
                    now = datetime.datetime.now().isoformat(' ')
                    headers = {HEADER_NAME_DATE: now}


                data.pop('Timestamp', None)

                # if a string then topics are string path
                # using device path and the data point.
                if isinstance(unit, str):
                    # publish the all point
                    self._publish_point(device_root, 'all', data)
                else:
                    # dictionary of "all" level containers.
                    all_publish = {}
                    # Loop over data from the csv file.
                    for sensor, value in data.items():
                        # if header_point_map isn't described then
                        # we are going to attempt to publish all of hte
                        # points based upon the column headings.
                        if not header_point_map:
                            if value:
                                sensor_name = sensor.split('/')[-1]
                                # container should start with a /
                                container = '/' + '/'.join(sensor.split('/')[:-1])                               
                                
                                publish_point(device_root+container,
                                                      sensor_name, value)
                                if container not in all_publish.keys():
                                    all_publish[container] = {}
                                all_publish[container][sensor_name] = value
                        else:
                            # Loop over mapping from the config file
                            for prefix, container in header_point_map.items():
                                if sensor.startswith(prefix):
                                    try:
                                        _, sensor_name = sensor.split('_')
                                    except:
                                        sensor_name = sensor
                                    
                                    # make sure that there is an actual value not
                                    # just an empty string.
                                    if value:
                                        if value == '0.0':
                                            pass
                                        # Attempt to publish as a float.
                                        try:
                                            value = float(value)
                                        except:
                                            pass
                                        
                                        publish_point(device_root+container,
                                                      sensor_name, value)
        
                                        if container not in all_publish.keys():
                                            all_publish[container] = {}
                                        all_publish[container][sensor_name] = value
    
                                    # move on to the next data point in the file.
                                    break

                    for _all, values in all_publish.items():
                        publish_point(device_root, _all+"/all", values)



        @Core.receiver('onfinish')
        def finish(self, sender):
            if self._src_file_handle is not None:
                try:
                    self._src_file_handle.close()
                except Exception as e:
                    self._log.error(e.message)

    Publisher.__name__ = 'DataPub'
    return Publisher(**kwargs)


def main(argv=sys.argv):
    '''Main method called by the eggsecutable.'''
    utils.vip_main(DataPub, version=__version__)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
