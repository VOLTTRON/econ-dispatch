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
import logging

import numpy as np
import pandas as pd
import scipy.stats as st
from pandas.tseries.holiday import Holiday
from dateutil.relativedelta import MO, TH

from econ_dispatch.forecast_models.history import Forecast as HistoryForecastBase


LOG = logging.getLogger(__name__)

def my_import(name):
    """Import submodule by name
    
    :param name: full module name, e.g., sklearn.linear_models.Ridge
    :returns: submodule, e.g., Ridge
    """
    components = name.split('.')
    mod = __import__(components[0], globals(), locals(), components[1:], -1)
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def make_time_features(ts, index=None, epoch=None, epoch_span=None):
    """Project datetimes into vector space for use in machine learning models

    Outputs: 
    
    - projection onto the unit circle of
      - second of day
      - day of week
      - day of year
    - seconds since `epoch`, normalized by `epoch_span`
    - binary workday indicator (i.e., Monday-Friday except major US holidays)
    
    :param ts: timestamp(s) to process
    :type ts: datetime.datetime or iterable thereof
    :param index: index of ts (e.g., if from a larger dataframe)
    :param epoch: start of time reckoning
    :param epoch_span: length of time reckoning
    :rtype: pd.DataFrame
    :returns: various projections of datetimes into vector space
    """
    # input validation
    try:
        if len(ts) == 1:
            _singleton = True
        elif len(ts) > 1:
            _singleton = False
        elif len(ts) < 1:
            raise ValueError("must pass non-empty iterable of timestamps")
    except TypeError:
        return make_time_features([ts], index=index, epoch=epoch, epoch_span=epoch_span)

    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.Series(0, index=ts).index
    if not isinstance(ts, pd.DatetimeIndex):
        raise ValueError("must pass non-empty iterable of timestamps")

    if index is None:
        index = pd.RangeIndex(len(ts))
    if epoch is None:
        epoch = min(ts)
    if epoch_span is None:
        epoch_span = float((end - epoch).total_seconds())

    time_features = {}
    start = min(ts)
    end = max(ts)

    # Major US holidays
    NewYearsDay = pd.tseries.holiday.Holiday('New Years Day', month=1, day=1)
    MemorialDay = pd.tseries.holiday.Holiday('Memorial Day', month=6, day=1, offset=pd.DateOffset(weekday=MO(-1)))
    IndependenceDay = pd.tseries.holiday.Holiday('Independence Day', month=7, day=4)
    LaborDay = pd.tseries.holiday.Holiday('Labor Day', month=9, day=1, offset=pd.DateOffset(weekday=MO(1)))
    ThanksgivingDay = pd.tseries.holiday.Holiday('Thanksgiving Day', month=11, day=1, offset=pd.DateOffset(weekday=TH(4)))
    ChristmasDay = pd.tseries.holiday.Holiday('Christmas Day', month=12, day=25)
    holidays = \
        NewYearsDay.dates(start.date(), end.date()).tolist() +\
        MemorialDay.dates(start.date(), end.date()).tolist() +\
        IndependenceDay.dates(start.date(), end.date()).tolist() +\
        LaborDay.dates(start.date(), end.date()).tolist() +\
        ThanksgivingDay.dates(start.date(), end.date()).tolist() +\
        ChristmasDay.dates(start.date(), end.date()).tolist()
    holidays = set([h.date() for h in holidays])

    # projections onto unit circle
    time_features['day_cos'] = np.cos((ts.hour * 3600 + ts.minute * 60 + ts.second) * 2 * np.pi / 86400.)
    time_features['day_sin'] = np.sin((ts.hour * 3600 + ts.minute * 60 + ts.second) * 2 * np.pi / 86400.)
    time_features['week_cos'] = np.cos(ts.dayofweek * 2 * np.pi / 7.)
    time_features['week_sin'] = np.sin(ts.dayofweek * 2 * np.pi / 7.)
    time_features['year_cos'] = np.cos(ts.dayofyear * 2 * np.pi / 365.)
    time_features['year_sin'] = np.sin(ts.dayofyear * 2 * np.pi / 365.)
    # linear march through time
    time_features['epoch'] = (ts - epoch).total_seconds() / epoch_span
    # workday indicator
    time_features['workday'] = [int(weekday < 5 and date not in holidays) for weekday, date in zip(ts.weekday, ts.date)]

    if _singleton:
        return {k: v[0] for k, v in time_features.iteritems()}
    else:
        return pd.DataFrame(time_features, index=index)


class Forecast(HistoryForecastBase):
    """Return forecasts from scikit-learn regression on historical data

    :param dependent_variables: historical variables to regress on
    :param model_name: name of module with scikit-learn regression interface
    :param model_settings: keyword arguments for model
    :param kwargs: keyword arguments for base class
    """
    def __init__(self,
                 dependent_variables=[],
                 model_name='sklearn.linear_models.Ridge',
                 model_settings={},
                 **kwargs):
        super(Forecast, self).__init__(**kwargs)
        if isinstance(dependent_variables, str):
            dependent_variables = [dependent_variables]
        self.dependent_variables = dependent_variables
        self.model = my_import(model_name)(**model_settings)

        self.independent_variables = []
        self.use_timestamp = False
        self.epoch = None
        self.epoch_span = None

    def train(self, training_data):
        """Train regression model on historical data

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        # load and preprocess
        super(Forecast, self).train(training_data)
        # remove NaNs
        self.historical_data = self.historical_data.loc[~self.historical_data.isnull().any(axis=1)]
        # project timestamps into vector space
        if self.timestamp_column in self.historical_data.columns:
            self.use_timestamp = True
            ts = self.historical_data.set_index(self.timestamp_column).index
            self.epoch = min(ts)
            self.epoch_span = float((max(ts) - self.epoch).total_seconds())
            time_features = make_time_features(ts,
                                               index=self.historical_data.index,
                                               epoch=self.epoch,
                                               epoch_span=self.epoch_span)
            self.historical_data = pd.concat([self.historical_data, time_features], axis=1)
            self.historical_data.drop(self.timestamp_column, axis=1, inplace=True)
        # leave all other variables independent
        self.independent_variables = [name for name in self.historical_data.columns
                                      if name not in self.dependent_variables]

        self.model.fit(self.historical_data[self.independent_variables],
                       self.historical_data[self.dependent_variables])

        # release historical data to save on memory
        # note that python garbage collection is not instantaneous
        LOG.warn("Releasing building load forecast training data. The agent will not be able to retrain on this data")
        self.historical_data = None

    def derive_variables(self, now, weather_forecast={}):
        """Predict forecast using regression model

        :param now: time of forecast
        :type now: datetime.datetime
        :param weather_forecast: dict containing a weather forecast
        :returns: dict of forecasts for time `now`
        """
        # project timestamps into vector space
        if self.use_timestamp:
            time_features = make_time_features(
                now, epoch=self.epoch, epoch_span=self.epoch_span)
            weather_forecast.update(time_features)
        X = pd.DataFrame(weather_forecast, index=[0])

        # Only ever see one record a time: pop values from 2D array
        y = self.model.predict(X[self.independent_variables])[0]
        result = {k: v for k, v in zip(self.dependent_variables, y)}
        return result
