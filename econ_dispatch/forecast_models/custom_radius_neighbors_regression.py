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
import logging

import numpy as np
import pandas as pd
import scipy.stats as st

from econ_dispatch.forecast_models.history import Forecast as HistoryForecastBase

LOG = logging.getLogger(__name__)


class Forecast(HistoryForecastBase):
    """Return forecasts and uncertainty estimate from custom regression on
    historical data

    The regression is based on the "radius neighbors" paradigm: return the
    average of data points in some fixed-size neighborhood around a query.

    We adapt this to time inputs by considering historical data within some
    time window from the same date in historical years, and modulo weekday.

    If too few data points are found within the neighborhood, all its radii
    are doubled. If this happens too many times, we use the global average over
    all historical data.

    Either means or medians may be used as an average.

    The method's main advantage is that it comes equipped with a measure of
    uncertainty: we have samples from the historical distribution, and so can
    calculate a tolerance interval around our population estimate. Here,
    we assume a Gaussian prior and compute a 1-sided tolerance interval
    (i.e., an upper bound), parameterized by the likelihood of error and the
    proportion of the distribution to be less than the upper bound estimate.

    :param time_diff_tolerance: radius around query datetime in which to
        look for data (in whole hours, modulo weekday/weekend)
    :param independent_variable_tolerances: radius around each non-datetime
        independent variable in which to look for data
    :param min_samples: fewer samples will trigger an increase in radii
    :param max_retries: number of times to increase radii
    :param scale_method: one of 'MAE' (mean absolute error) or 'MSE'
        (mean squared error)
    :param alpha: likelihood of error in upper bound calculation
    :param p: proportion of samples to be less than calculated upper bound
    :param kwargs: keyword arguments for base class
    """

    def __init__(
        self,
        time_diff_tolerance=1,
        independent_variable_tolerances={},
        min_samples=2,
        max_retries=5,
        scale_method="MSE",
        alpha=0.05,
        p=0.99,
        **kwargs,
    ):
        super(Forecast, self).__init__(**kwargs)
        self.time_diff_tolerance = time_diff_tolerance
        if int(self.time_diff_tolerance) != self.time_diff_tolerance:
            time_diff_tolerance = int(time_diff_tolerance) + 1
            LOG.debug(
                "Time difference tolerance must be in whole hours. "
                "Rounding up from {} to {}".format(self.time_diff_tolerance, time_diff_tolerance)
            )
            self.time_diff_tolerance = time_diff_tolerance
        self.independent_variable_tolerances = independent_variable_tolerances
        self.min_samples = min_samples
        self.max_retries = max_retries
        self.scale_method = scale_method
        self.alpha = alpha
        self.p = p

        self.num_retries = 0

    def derive_variables(self, now, weather_forecast={}):
        """Predict forecast using regression model

        :param now: time of forecast
        :type now: datetime.datetime
        :param weather_forecast: dict containing a weather forecast
        :returns: dict of forecasts for time `now`
        """
        day_of_week = now.weekday()
        hour_of_day = now.hour
        df = self.historical_data

        # requires minimum difference in timestamps >= 1min
        filter = (
            (df[self.timestamp_column].dt.weekday == day_of_week)
            & (df[self.timestamp_column].dt.hour >= hour_of_day - self.time_diff_tolerance)
            & (
                (df[self.timestamp_column].dt.hour < hour_of_day + self.time_diff_tolerance)
                | (
                    (df[self.timestamp_column].dt.hour == hour_of_day + self.time_diff_tolerance)
                    & (df[self.timestamp_column].dt.minute == 0)
                )
            )
        )
        if hour_of_day > 23 - self.time_diff_tolerance:
            # last hour to include in the next day
            delta = (hour_of_day + self.time_diff_tolerance) % 24
            filter |= (df[self.timestamp_column].dt.weekday == (day_of_week + 1) % 7) & (
                df[self.timestamp_column].dt.hour < delta
            )
            filter |= (
                (df[self.timestamp_column].dt.weekday == (day_of_week + 1) % 7)
                & (df[self.timestamp_column].dt.hour == delta)
                & (df[self.timestamp_column].dt.minute == 0)
            )
        if hour_of_day < 0 + self.time_diff_tolerance:
            # first hour to include in the previous day
            delta = (hour_of_day - self.time_diff_tolerance) % 24
            filter |= (df[self.timestamp_column].dt.weekday == (day_of_week - 1) % 7) & (
                df[self.timestamp_column].dt.hour >= delta
            )

        for variable, value in weather_forecast.items():
            tolerance = self.independent_variable_tolerances.get(variable, None)
            if tolerance is None:
                continue
            # Note: this will also filter any records where
            # independent variable is NaN
            filter &= df[variable] >= value - tolerance
            filter &= df[variable] <= value + tolerance

        filtered_data = df[filter]
        min_num_samples = filtered_data.shape[0] - sum(pd.isnull(filtered_data).any(axis=1))

        if min_num_samples >= self.min_samples:
            if self.num_retries > 0:
                self.adjust_tolerances(1.0 / 2 ** self.num_retries)
                self.num_retries = 0
            return self.build_results(filtered_data)
        else:
            if self.num_retries == self.max_retries:
                LOG.debug(
                    "Maximum number of tolerance increases {} reached. "
                    "Defaulting to all-of-data.".format(self.max_retries)
                )
                self.adjust_tolerances(1.0 / 2 ** self.num_retries)
                self.num_retries = 0
                return self.build_results(df)
            else:
                self.adjust_tolerances(2)
                self.num_retries += 1
                LOG.debug(
                    "Number of samples {} is less than the minimum {} "
                    "after {} retries. Time tolerance increased to {} "
                    " and other tolerances increased to {}".format(
                        min_num_samples,
                        self.min_samples,
                        self.num_retries,
                        self.time_diff_tolerance,
                        self.independent_variable_tolerances,
                    )
                )
                return self.derive_variables(now, weather_forecast)

    def build_results(self, data):
        """Build tolerance intervals from historical data

        :param data: pre-filtered historical data
        :returns: average and upper bound for each dependent variable
        """
        results = {}
        for dep in self.dependent_variables:
            # NaN values in dependent variables usually co-occur with
            # NaNs in indpendent variables, so are already filtered.
            # Filter them anyway.
            loc, (_, ub) = self.estimate_center(data[dep].loc[~pd.isnull(data[dep])])
            results[dep] = loc
            results[dep + "_ub"] = ub
        return results

    def adjust_tolerances(self, factor):
        """Change neighborhood radii

        :param factor: multiply all radii by this number
        """
        self.time_diff_tolerance *= factor
        for k in self.independent_variable_tolerances:
            self.independent_variable_tolerances[k] *= factor

    def get_loc_scale(self, a, scale_method="MAD"):
        """Compute measure of central tendency and spread

        :param a: data
        :param scale_method: one of 'MAD' (median absolute distance) or
            'MSE' (mean squared error) for maeasuring spread; the minimum
            of this metric is chosen as the center (median and mean,
            respectively)
        :returns: estimates of location and scale
        :rtype: tuple[float]
        """
        if len(a) < 2:
            LOG.warning("Not enough samples to estimate " "distribution parameters.")
            return a[0], 0
        if scale_method == "MAD":
            _c = 0.6744897501960817  # st.norm.ppf(3/4)
            med = np.median(a)
            return med, np.median(np.absolute(med - a)) / _c
        if scale_method == "MSE":
            return np.mean(a), np.std(a)
        LOG.debug("Location and scale estimation method {} " "not found; defaulting to MSE".format(scale_method))
        return self.get_loc_scale(a, "MAD")

    def tolerance_interval(self, x, alpha=0.05, p=0.99, loc=None, scale=None):
        """Calculate one-sided normal tolerance interval as in
            https://www.jstatsoft.org/article/view/v036i05. With probability
            1-alpha, 100*p percent of new samples will be less than the upper
            bound (or greater than lower bound, because Gaussian is symmetric).

        :param x: data, assumed iid and normal
        :param alpha: tolerance of p-percentile
        :param p: proportion of data to be covered by tolerance interval
        :param loc: mean of Gaussian prior
        :param scale: standard deviation of Gaussian prior
        :returns: estimates of tolerance bounds
        :rtype: tuple(float)
        """
        n = len(x)
        if n < 2:
            LOG.warning("Not enough samples to estimate " "distribution parameters.")
            return x[0], (-np.inf, np.inf)
        df = n - 1

        if loc is None:
            loc = x.mean()
        if scale is None:
            scale = x.std()

        z_p = st.norm.ppf(1 - p, loc=loc, scale=scale)
        ncp = np.sqrt(n) * z_p
        t_a = st.nct.ppf(1 - alpha, df, ncp)
        K = t_a / np.sqrt(n)

        return sorted((loc - scale * K, loc + scale * K))

    def estimate_center(self, data):
        """Estimate center and tolerance interval of 1D data

        :param data: load data
        :param scale_method: kwarg for get_loc_scale
        :param alpha: kwarg for tolerance_interval
        :param p: kwarg for tolerance_interval
        :returns: estimates of location and tolerance bounds
        :rtype: tuple[float]
        """
        loc, scale = self.get_loc_scale(data, scale_method=self.scale_method)
        _data = (data - loc) / scale  # stabilize numerical errors
        lb, ub = self.tolerance_interval(_data, self.alpha, self.p, loc=0, scale=1)
        return loc, (lb * scale + loc, ub * scale + loc)
