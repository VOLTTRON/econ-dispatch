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
import datetime
import logging

import pandas as pd
import pytz
import requests
from volttron.platform.agent import utils

from econ_dispatch.forecast_models import ForecastBase
from econ_dispatch.utils import round_to_hour

LOG = logging.getLogger(__name__)

LIVE_URL_TEMPLATE = "https://api.weather.gov/points/{latitude},{longitude}/forecast/hourly"
KEYS = {"temperature": "temperature", "windSpeed": "wind_speed", "windDirection": "wind_direction"}
TIMESTAMP_FIELD = "timestamp"


class Weather(ForecastBase):
    """Return weather forecast from National Weather Service web API

    :param latitude: latitudinal coordinates of location
    :param longitude: longitudinal coordinates of location
    :param timezone: local timezone of location (NWS API uses local time)
    :param hours_forecast: how long the forecast should be. Note that NWS
        provides only hourly data
    :param kwargs: kwargs for `forecast_models.ForecastBase`
    """

    def __init__(self, latitude=None, longitude=None, timezone="UTC", hours_forecast=24, **kwargs):
        super(Weather, self).__init__(**kwargs)
        self.url = LIVE_URL_TEMPLATE.format(latitude=latitude, longitude=longitude)
        assert hours_forecast <= 156, "NWS returns a maximum of 6.5 days"
        self.hours_forecast = hours_forecast
        self.timezone = pytz.timezone(timezone)

    def derive_variables(self, now):
        pass

    def get_weather_forecast(self, now):
        """Validate and return weather forecasts

        :param now: timestamp of first hour
        :type now: datetime.datetime
        """
        now = now.astimezone(self.timezone)
        start = round_to_hour(now)
        hours = [start + datetime.timedelta(hours=i) for i in range(self.hours_forecast)]

        results = self.get_live_data()

        # Join to desired timestamps
        results = pd.DataFrame(results).set_index(TIMESTAMP_FIELD).T
        _empty = pd.DataFrame(index=pd.Index(hours, name=TIMESTAMP_FIELD))
        results = _empty.join(results.T, how="left")

        # Fill missing data
        missing_hours = results.loc[results.isna().any(axis=1)].index.tolist()
        if len(missing_hours) == 24:
            raise ValueError(
                "Weather forecast has no overlap with requested hours. Should you use historical data instead?"
            )
        elif missing_hours:
            LOG.warning(f"Weather forecast does not contain values for {missing_hours}. Filling in.")
            results = results.bfill().ffill()

        # cast to expected format
        _results = []
        for ts, vals in results.iterrows():
            vals = vals.to_dict()
            vals.update({TIMESTAMP_FIELD: ts.to_pydatetime()})  # pd.Timestamp --> datetime.datetime
            _results.append(vals)
        results = _results

        LOG.debug("Weather forecast from {} to {}".format(results[0][TIMESTAMP_FIELD], results[-1][TIMESTAMP_FIELD]))
        return results

    def get_live_data(self):
        """Query and parse NWS records"""
        r = requests.get(self.url)
        try:
            r.raise_for_status()
            parsed_json = r.json()
            records = parsed_json["properties"]["periods"]
        except (requests.exceptions.HTTPError, ValueError, KeyError) as e:
            LOG.error("Error retrieving weather data: " + str(e))
            raise e

        results = []
        for rec in records[: self.hours_forecast]:
            timestamp = utils.parse_timestamp_string(rec["endTime"])
            timestamp = timestamp.astimezone(pytz.UTC)
            result = {TIMESTAMP_FIELD: timestamp}
            result.update(self.get_nws_forecast_from_record(rec))
            results.append(result)
        return results

    def get_nws_forecast_from_record(self, record):
        """Parse single NWS record"""
        result = {}
        for key, value in KEYS.items():
            try:
                result[KEYS[key]] = float(record[key])
            except ValueError:
                value = record[key].split()
                if len(value) == 2:
                    # result[KEYS[key]+"_unit"] = value[1]
                    value = float(value[0])
                    result[KEYS[key]] = value
                else:
                    result[KEYS[key]] = record[key]
            except KeyError:
                LOG.error("Weather record did not contain {}".format(key))
        return result
