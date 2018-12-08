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

from econ_dispatch.forecast_models import ForecastModelBase
from HTMLParser import HTMLParser
from dateutil import parser
import requests
import datetime

NG_DATA = {"tomorrow_month":"",
          "tomorrow_day":"",
          "tomorrow_year":"",
          "minimum_month":"",
          "minimum_day":"",
          "minimum_year":"",
          "load_area":"3",
          "svc_class":"SC-3 HP",
          "volt_level":"2",
          "from_month":"7",
          "from_day":"15",
          "from_year":"2010",
          "to_month":"7",
          "to_day":"20",
          "to_year":"2010",
          "Display":"TEXT",
          "la":"3",
          "vdl":"2",
          "sc":"SC-3 HP",
          "fd":"07152010",
          "td":"07202010"}

POST_URL = 'https://www9.nationalgridus.com/niagaramohawk/business/rates/5_hour_charge_a.asp'

MAX_UPDATE_FREQUENCY = datetime.timedelta(hours=0.5)

def round_to_hour(dt):
    dt_start_of_hour = dt.replace(minute=0, second=0, microsecond=0)
    dt_half_hour = dt.replace(minute=30, second=0, microsecond=0)

    if dt >= dt_half_hour:
        # round up
        dt = dt_start_of_hour + datetime.timedelta(hours=1)
    else:
        # round down
        dt = dt_start_of_hour

    return dt

class Parser(HTMLParser):
    def __init__(self, *args, **kwargs):
        HTMLParser.__init__(self, *args, **kwargs)
        self.results = {}
        self.reading_table = False
        self.current_date_str = None
        self.current_date_time = None

    def handle_starttag(self, tag, attrs):
        if tag == "table" and ("id", "Table3") in attrs:
            self.reading_table = True

    def handle_endtag(self, tag):
        if tag == "table" and self.reading_table == True:
            self.reading_table = False

    def handle_data(self, data):
        if self.reading_table:
            if ":" in data:
                hour, minute = data.split(":")
                hour = int(hour) - 1
                minute = int(minute)

                self.current_date_time = parser.parse(self.current_date_str + " {}:{}".format(hour, minute),
                                                      dayfirst=False)
            else:
                try:
                    f = float(data)
                    self.results[self.current_date_time] = f
                except ValueError:
                    pass
        elif data.startswith("Prices for: "):
            date_string = data[-10:]
            self.current_date_str = date_string

class Model(ForecastModelBase):
    def __init__(self, load_area=3, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.load_area = str(load_area)
        self.last_collection = None
        self.values = {}
        self.last_value = None

    def derive_variables(self, now, independent_variable_values={}):
        """Get the predicted load values based on the independent variables."""
        self.update_values(now)

        rounded_time = round_to_hour(now)

        for _ in xrange(3):
            if rounded_time in self.values:
                break
            rounded_time = rounded_time - datetime.timedelta(days=1)

        return {"electricity_cost": self.values.get(rounded_time,
                                                    self.last_value)}

    def get_request_params(self, start, end):
        data = NG_DATA.copy()
        data["from_month"] = str(start.month)
        data["from_day"] = str(start.day)
        data["from_year"] = str(start.year)

        data["to_month"] = str(end.month)
        data["to_day"] = str(end.day)
        data["to_year"] = str(end.year)

        data["fd"] = start.strftime("%m%d%Y")
        data["td"] = end.strftime("%m%d%Y")

        data["la"] = data["load_area"] = self.load_area

        return data

    def update_values(self, now):
        if (self.last_collection is not None and
        (now - self.last_collection) < MAX_UPDATE_FREQUENCY):
            return

        today = now.date()
        tomorrow = today + datetime.timedelta(days=1)
        yesterday = today - datetime.timedelta(days=1)

        request_params = self.get_request_params(yesterday, tomorrow)

        r = requests.post(POST_URL, data=request_params)

        html_parser = Parser()
        html_parser.feed(r.text)

        self.values = html_parser.results
        self.last_value = self.values[max(self.values.iterkeys())]

        self.last_collection = now
