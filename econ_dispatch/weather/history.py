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

from dateutil.parser import parse
import logging
_log = logging.getLogger(__name__)
import pandas as pd
import datetime as dt
from econ_dispatch import utils

time_step = dt.timedelta(hours=1)

from econ_dispatch.forecast_models import HistoryModelBase

class Weather(HistoryModelBase):
    def __init__(self, hours_forecast=24, history_data={}, **kwargs):
        super(Weather, self).__init__(**kwargs)
        self.hours_forecast = hours_forecast
        training_data = utils.normalize_training_data(history_data)
        self.train(training_data)

    def get_weather_forecast(self, now):
        # The Solar Radiation model needs the current weather data as
        # well as data from 24 hours ago. We return a dictionary that
        # has both. Keys for yesterday's data are suffixed with -24

        def merge_results(today, yesterday):
            for k, v in yesterday.items():
                k = k + "-24"
                today[k] = v
            return today

        results_current = self.get_historical_data(now)
        results_minus24h = self.get_historical_data(now - dt.timedelta(hours=24))

        results = [merge_results(t, y) for t, y in zip(results_current, results_minus24h)]

        return results

    def get_historical_data(self, now):
        now = now.replace(year=self.history_year)

        results = []
        for _ in xrange(self.hours_forecast):
            record = self.get_historical_hour(now)
            record["timestamp"] = now
            results.append(record)
            now += time_step
            now = now.replace(year=self.history_year)

        return results
