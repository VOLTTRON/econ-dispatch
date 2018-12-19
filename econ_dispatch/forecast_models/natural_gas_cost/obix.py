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
import requests
import xml.etree.ElementTree as ET
from econ_dispatch.forecast_models import ForecastModelBase
import pytz

import logging

_log = logging.getLogger(__name__)

MAX_UPDATE_FREQUENCY = datetime.timedelta(hours=0.5)

class Model(ForecastModelBase):
    def __init__(self, url="http://becchp.com/obix/histories/BECHP",
                 user_name = None,
                 password = None,
                 point_name="Nat Gas Com Cost per MM-BTU"):

        self.point_name = point_name
        interface_point_name = point_name.replace(" ", "$20").replace("-", "$2d")
        url = url if url.endswith("/") else url + "/"

        self.url = url + interface_point_name + '/~historyQuery'
        self.user_name = user_name
        self.password = password
        self.last_value = None
        self.last_collection = None

    def derive_variables(self, now, independent_variable_values={}):
        """Get the predicted load values based on the independent variables."""
        self.update_value(now)

        return {"natural_gas_cost": self.last_value}

    def time_format(self, dt):
        """Format timestamp for becchp.com query"""
        return u"%s:%06.3f%s" % (
            dt.strftime('%Y-%m-%dT%H:%M'),
            float("%.3f" % (dt.second + dt.microsecond / 1e6)),
            dt.strftime('%z')[:3] + ':' + dt.strftime('%z')[3:]
        )

    def update_value(self, now):
        if (self.last_collection is not None and
        (now - self.last_collection) < MAX_UPDATE_FREQUENCY):
            return

        end_time = pytz.UTC.localize(now) + datetime.timedelta(days=1)
        start_time = end_time - datetime.timedelta(days=8)

        print self.url

        # becchp.com does not accept percent-encoded parameters
        # requests is not configurable to not encode (from lead dev: https://stackoverflow.com/a/23497903)
        # do it manually:
        payload = {'start': self.time_format(start_time),
                   'end': self.time_format(end_time)}
        payload_str = "&".join("%s=%s" % (k, v) for k, v in payload.items())

        r = requests.get(self.url, auth=(self.user_name, self.password), params=payload_str)

        print r.url

        try:
            r.raise_for_status()
            print r.text
            result = self.parse_result(r.text)
            if result is not None:
                self.last_value = result
        except StandardError as e:
            print repr(e)
            return

    def parse_result(self, xml_tree):
        obix_types = {'int': int,
                      'bool': bool,
                      'real': float}
        obix_schema_spec = '{http://obix.org/ns/schema/1.0}'
        root = ET.fromstring(xml_tree)
        value_def = root.findall(".{0}obj[@href='#RecordDef']*[@name='value']".format(obix_schema_spec))

        if len(value_def) == 0:
            _log.error("No values in time slice")
            return None
        elif len(value_def) > 1:
            _log.error("xml does not match obix standard schema")
            return None
        else:
            value_def = value_def[0]

        value_tag = value_def.tag[len(obix_schema_spec):]

        records = root.findall(".{0}list/".format(obix_schema_spec))

        values = [obix_types[value_tag](record.find("./*[@name='value']").attrib['val']) for record in records]

        return values[-1] if values else None