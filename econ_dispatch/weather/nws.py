# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2018, Battelle Memorial Institute
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

import requests
import logging

from volttron.platform.agent import utils

_log = logging.getLogger(__name__)

live_url_template = "https://api.weather.gov/points/{latitude},{longitude}/forecast/hourly"
KEYS = {
    "temperature": "temperature",
    "windSpeed": "wind_speed",
    "windDirection": "wind_direction"
}

class Weather(object):
    def __init__(self, latitude=None, longitude=None, hours_forecast=24):
        self.latitude = latitude
        self.longitude = longitude
        self.hours_forecast = hours_forecast
        #  assert hours_forecast <= 156

    def get_weather_forecast(self, now):
        results = self.get_live_data()

        return results
    
    def get_live_data(self):
        url = live_url_template.format(latitude=self.latitude, longitude=self.longitude)
        r = requests.get(url)
        try:
            r.raise_for_status()
            parsed_json = r.json()
        except (requests.exceptions.HTTPError, ValueError) as e:
            _log.error("Error retrieving weather data: " + str(e))
            return []
        
        results = []
        records = parsed_json["properties"]["periods"]
        for rec in records[:self.hours_forecast]:
            result = {"timestamp": utils.parse_timestamp_string(rec["endTime"])}
            result.update(self.get_nws_forecast_from_record(rec))
            results.append(result)
        return results
    
    def get_nws_forecast_from_record(self, record):
        result = {}
        for key, value in KEYS.iteritems():
            try:
                result[KEYS[key]] = float(record[key])
            except ValueError:
                value = record[key].split()
                if len(value) == 2:
                    value, unit = float(value[0]), value[1]
                    result[KEYS[key]] = value
                    # result[KEYS[key]+'_unit'] = unit
                else: 
                    result[KEYS[key]] = record[key]
            except KeyError as e:
                _log.error("Weather record did not contain {}".format(key))
        return result