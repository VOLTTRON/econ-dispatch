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
from collections import defaultdict
import requests
import datetime

NG_DATA = {"tomorrow_month":"",
          "tomorrow_day":"",
          "tomorrow_year":"",
          "minimum_month":"",
          "minimum_day":"",
          "minimum_year":"",
          "load_area":"3",
          "svc_class":"all",
          "volt_level":"all",
          "from_month":"7",
          "from_day":"15",
          "from_year":"2010",
          "to_month":"7",
          "to_day":"20",
          "to_year":"2010",
          "Display":"TEXT",
          "la":"3",
          "vdl":"all",
          "sc":"all",
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
        self.results = defaultdict(dict)
        self.rows = []
        self.reading_table = False
        self.reading_row = False
        self.reading_cell = False
        self.current_date_str = None
        self.current_date_time = None
        self.current_row = []
        self.header_row = None

    def handle_starttag(self, tag, attrs):
        if tag == "table" and ("id", "Table5") in attrs:
            self.reading_table = True
        if tag == "tr" and self.reading_table:
            self.reading_row = True
        if tag == "td" and self.reading_row:
            self.reading_cell = True


    def handle_endtag(self, tag):
        if tag == "table":
            if self.reading_table:
                self.build_results()
            self.reading_table = False
        if tag == "tr":
            self.reading_row = False
            if self.reading_table:
                self.finish_row()
        if tag == "td":
            self.reading_cell = False

    def finish_row(self):
        if self.header_row is None:
            self.header_row = self.current_row
        else:
            self.rows.append(self.current_row)
        self.current_row = []

    def handle_data(self, data):
        if self.reading_cell:
            self.current_row.append(data.strip())

    def build_results(self):
        for row in self.rows:
            record = dict(zip(self.header_row, row))
            hour, minute = record["Hour Ending"].split(":")
            hour = int(hour) - 1
            time_stamp = parser.parse(record["Date"] + " {}:{}".format(hour, minute), dayfirst=False)

            service_class = "sc3a" if record["Service Class"] == "SC3A" else "sc3"

            label = (record["Load Area"].lower() + "_" +
                     service_class + "_" +
                     record["Voltage Level"].lower())

            value = float(record["$/MWh"])

            self.results[time_stamp][label] = value

        self.results = dict(self.results)

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

        return self.values.get(rounded_time, self.last_value)

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
