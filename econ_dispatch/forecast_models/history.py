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
from collections import Counter
from datetime import timedelta
import logging

import pandas as pd
import numpy as np

from econ_dispatch.forecast_models import ForecastBase
from econ_dispatch.utils import preprocess


LOG = logging.getLogger(__name__)


class Forecast(ForecastBase):
    """Return historical data from a CSV

    :param timestamp_column: name of timestamp column in CSV
    :param preprocess_settings: see `econ_dispatch.utils` for defnition
    :param retain_old_on_retrain: whether to keep old data when training
        from historian
    :param threshold: number of hours around query time in which
        historical data is acceptable
    :param kwargs: keyword arguments for base class
    """
    def __init__(self,
                 timestamp_column="timestamp",
                 preprocess_settings=None,
                 retain_old_on_retrain=False,
                 threshold=0.5,
                 **kwargs):
        super(Forecast, self).__init__(**kwargs)
        self.timestamp_column = timestamp_column
        self.preprocess_settings = preprocess_settings
        self.retain_old_on_retrain = retain_old_on_retrain
        self.threshold = timedelta(hours=threshold)
        self.historical_data = None

    def derive_variables(self, now, weather_forecast={}):
        """Return record with closest matching date
        
        If no date is found within `self.threshold` hours, try changing the
        year to match historical data

        :param now: time of forecast
        :type now: datetime.datetime
        :param weather_forecast: dict containing a weather forecast
        :returns: dict of forecasts for time `now`
        """
        if self.historical_data is None:
            raise ValueError("Forecast model not trained.")
        # make pandas.DatetimeIndex for faster datetime function calls
        times = self.historical_data[self.timestamp_column]
        times = pd.Series(
            data=np.empty(times.shape),
            index=times).index

        # assume both are localized to UTC
        indices = abs(times - now)

        if min(indices) > self.threshold:
            # query datetime not in historical data. try to find same date in a
            # different year (sort years by number of records to speed search)
            counts = Counter(times.year)
            years = sorted(list(counts.keys()), key=lambda k: -1*counts[k])
            for year in years:
                _now = now.replace(year=year)
                indices = abs(times - _now)
                if min(indices) <= self.threshold:
                    break
            else:
                raise ValueError("No historical data within {} hours of query "
                                 "datetime {}, irrespective of year"
                                 "".format(self.threshold, now))

        index = pd.Series(indices).idxmin()
        # Return the row as a dict.
        return dict(self.historical_data.iloc[index])

    def train(self, training_data):
        """Load and preprocess historical data

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        # build dataframe
        results = {}
        for key, values in training_data.iteritems():
            readings = pd.Series(values)
            results[key] = readings
        df = pd.DataFrame(results)
        # localize to UTC
        if self.timestamp_column is not None:
            df[self.timestamp_column] = pd.to_datetime(
                df[self.timestamp_column])
            try:
                df[self.timestamp_column] = df[self.timestamp_column].apply(
                    lambda ts: ts.tz_localize('UTC'))
            except TypeError:
                df[self.timestamp_column] = df[self.timestamp_column].apply(
                    lambda ts: ts.astimezone('UTC'))

        # keep old data if desired
        if self.retain_old_on_retrain and (self.historical_data is not None):
            df = df.set_index(self.timestamp_column, drop=True)
            hist_df = self.historical_data.set_index(self.timestamp_column,
                                                     drop=True)
            # Privilege new data in case of conflict
            cols_in = [j for j in hist_df.columns if j in df.columns]
            idx_in = [i for i in hist_df.index if i not in df.index]
            df = df.append(hist_df.loc[idx_in, cols_in], sort=True)
            # Retain old topics
            cols_out = [j for j in hist_df.columns if j not in df.columns]
            df[cols_out] = hist_df.loc[:, cols_out]

        # preprocess
        if self.preprocess_settings is not None:
            # override config so we never lose timestamps in preprocessing
            decision_variables = self.preprocess_settings.get('decision_variables')
            if decision_variables is not None:
                timestamp_vars = \
                    set(self.preprocess_settings.get('timezone', {}).keys())
                if self.timestamp_column is not None:
                    timestamp_vars.add(self.timestamp_column)
                for k in timestamp_vars:
                    if k not in decision_variables:
                        decision_variables.append(k)
                self.preprocess_settings['decision_variables'] = decision_variables

            df = preprocess(df, **self.preprocess_settings)

        # sort on time for faster retrieval
        if self.timestamp_column is not None:
            df.sort_values(self.timestamp_column)

        self.historical_data = df
