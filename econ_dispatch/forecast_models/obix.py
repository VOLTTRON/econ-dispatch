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
import datetime
import logging
import xml.etree.ElementTree as ET

from dateutil.parser import parse
import pandas as pd
import pytz
import requests

from econ_dispatch.forecast_models import ForecastBase
from econ_dispatch.utils import round_to_hour


LOG = logging.getLogger(__name__)

MAX_UPDATE_FREQUENCY = datetime.timedelta(hours=0.5)

class Forecast(ForecastBase):
    """Query a REST endpoint with an Obix backend

    :param url: REST endpoint
    :param point_name: Obix variable to query
    :param username: username for authentication
    :param password: password for authentication
    :param timezone: Obix server timezone
    :param dependent_variable: variable name for post-processing
    :param kwargs: keyword arguments for base class
    """
    def __init__(self,
                 url=None,
                 point_name="Nat Gas Com Cost per MM-BTU",
                 username=None,
                 password=None,
                 timezone='UTC',
                 dependent_variable='natural_gas_cost',
                 **kwargs):
        super(Forecast, self).__init__(**kwargs)
        self.url = url
        self.username = username
        self.password = password
        if self.url is None or self.username is None or self.password is None:
            raise ValueError("Obix url/username/password not set")

        self.point_name = point_name
        # Obix does not accept percent-encoded parameters
        # requests is not configurable to not encode
        # (https://stackoverflow.com/a/23497903)
        # do it manually:
        interface_point_name = \
            point_name.replace(" ", "$20").replace("-", "$2d")
        url = url if url.endswith("/") else url + "/"
        self.url = url + interface_point_name + '/~historyQuery'

        try:
            self.timezone = pytz.FixedOffset(int(timezone))
        except ValueError:
            self.timezone = pytz.timezone(timezone)

        self.dependent_variable = dependent_variable

        self.default_value = None  # latest successfully retrieved value
        self.last_collection = None

    def derive_variables(self, now, weather_forecast={}):
        """Retrieve values from Obix server. If not available, look back one day

        :param now: time of forecast
        :type now: datetime.datetime
        :param weather_forecast: dict containing a weather forecast
        :returns: dict of forecasts for time `now`
        """
        # Query for updates
        self.update_values(now)
        # Only look for data on the hour
        rounded_time = round_to_hour(now)
        date = rounded_time.astimezone(self.timezone).date()
        # Monday should look back to Friday, Sat ->, Sun -> Sun
        if date.weekday() == 0:
            lookback = 3
        elif date.weekday() < 5:
            lookback = 1
        else:
            lookback = 7

        if rounded_time in self.values:
            return {self.dependent_variable: self.values[rounded_time]}
        else:
            # look back one day
            new_time = rounded_time - datetime.timedelta(days=lookback)
            LOG.debug("{} for {} not available. "
                       "Trying {}".format(self.dependent_variable,
                                          rounded_time,
                                          new_time))
            rounded_time = new_time

        if rounded_time in self.values:
            return {self.dependent_variable: self.values[rounded_time]}
        else:
            # use latest succesfully retrieved value
            LOG.debug("{} for {} not available either. "
                       "Using default value".format(self.dependent_variable,
                                                    rounded_time))
            return {self.dependent_variable: self.default_value}

    def time_format(self, dt):
        """Format timestamp for Obix query"""
        return u"%s:%06.3f%s" % (
            dt.strftime('%Y-%m-%dT%H:%M'),
            float("%.3f" % (dt.second + dt.microsecond / 1e6)),
            dt.strftime('%z')[:3] + ':' + dt.strftime('%z')[3:]
        )

    def update_values(self, now):
        """Query Obix site for new values, but not often"""
        true_now = datetime.datetime.utcnow()
        true_now = pytz.UTC.localize(true_now)
        if (self.last_collection is not None and
                (true_now - self.last_collection) < MAX_UPDATE_FREQUENCY):
            return

        now = now.astimezone(self.timezone)
        end_time = now + datetime.timedelta(days=1)
        start_time = end_time - datetime.timedelta(days=8)

        # Obix does not accept percent-encoded parameters
        # requests is not configurable to not encode
        # (https://stackoverflow.com/a/23497903)
        # do it manually:
        payload = {'start': self.time_format(start_time),
                   'end': self.time_format(end_time)}
        payload_str = "&".join("%s=%s" % (k, v) for k, v in payload.items())

        r = requests.get(self.url,
                         auth=(self.username, self.password),
                         params=payload_str)

        try:
            r.raise_for_status()
        except StandardError as e:
            LOG.error(repr(e))
            return

        # assume next update will occur at the next 7am, local time
        _now = true_now.astimezone(self.timezone)
        next_update = _now.replace(hour=7,
                                   minute=0,
                                   second=0,
                                   microsecond=0)
        if _now.hour >= 7:
            next_update += datetime.timedelta(days=1)
        next_update = next_update.astimezone(pytz.UTC)

        values = self.parse_result(r.text, next_update)

        if len([k for k in values.iterkeys()]) == 0:
            LOG.debug("HTTP response is emtpy")
            return
        else:
            self.values = values
            # set default value to latest
            self.default_value = self.values[max(self.values.iterkeys())]
            self.last_collection = true_now

    def parse_result(self, xml_tree, next_update):
        """Parse XML response from Obix query"""
        obix_types = {'int': int,
                      'bool': bool,
                      'real': float}
        obix_schema_spec = '{http://obix.org/ns/schema/1.0}'
        root = ET.fromstring(xml_tree)
        value_def = root.findall(
            ".{0}obj[@href='#RecordDef']*[@name='value']".format(
                obix_schema_spec))

        if len(value_def) == 0:
            LOG.error("No values in time slice")
            return None
        elif len(value_def) > 1:
            LOG.error("xml does not match obix standard schema")
            return None
        else:
            value_def = value_def[0]

        value_tag = value_def.tag[len(obix_schema_spec):]

        records = root.findall(".{0}list/".format(obix_schema_spec))

        _times = [record.find("./*[@name='timestamp']").attrib['val']
                  for record in records]
        _values = [
            obix_types[value_tag](
                record.find("./*[@name='value']").attrib['val'])
            for record in records]

        _times = [round_to_hour(parse(_t)\
                  .replace(tzinfo=self.timezone)\
                  .astimezone(pytz.UTC)) for _t in _times]
        _values = [float(_v) for _v in _values]

        # pad times between updates with last value
        values = pd.Series(_values, index=_times)
        values = values.reindex(pd.date_range(start=values.index.min(),
                                              end=next_update,
                                              freq='1H'),
                                method='pad')

        return dict(values)
