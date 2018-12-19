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

import pandas as pd
import scipy.stats as st
import numpy as np
from econ_dispatch.forecast_models import ForecastModelBase

class Model(HistoryModelBase):
    def __init__(self, 
                 training_window=365,
                 training_sources={},
                 time_stamp_column="timestamp",
                 dependent_variables=[], 
                 time_diff_tolerance=1.0,
                 independent_variable_tolerances={}, 
                 **kwargs):
        # self.data_frame = pd.read_csv(training_csv, parse_dates=[time_stamp_column])
        super(Model, self).__init__(training_window=training_window, training_sources=training_sources)

        self.time_stamp_column = time_stamp_column
        self.dependent_variables = [str(d) for d in dependent_variables]
        self.independent_variable_tolerances = independent_variable_tolerances

        self.time_diff_tolerance = time_diff_tolerance
        if int(self.time_diff_tolerance) != self.time_diff_tolerance:
            time_diff_tolerance = int(time_diff_tolerance)+1
            _log.debug("Time difference tolerance must be in whole hours. Rounding up "
                       "from {} to {}".format(self.time_diff_tolerance, time_diff_tolerance))
            self.time_diff_tolerance = time_diff_tolerance

        self.min_samples = kwargs.get("min_samples", 2)
        self.max_retries = kwargs.get("max_retries", 5)
        self.scale_method = kwargs.get("scale_method", 'MSE')
        self.alpha = kwargs.get("alpha", 0.05)
        self.p = kwargs.get("p", 0.99)
        
        self.num_retries = 0
        self.historical_data = None
    
    
    def train(self, training_data):
        """ Add new data without losing old data"""
        if self.historical_data is None:
            super(Model, self).train(training_data)
            self.historical_data[time_stamp_column] = pd.to_datetime(self.historical_data[time_stamp_column])
        else:
            _log.warning("Not tested yet...historical data may be corrupted")
            results = {}
            for key, values in training_data.iteritems():
                readings = pd.Series(values)
                results[key] = readings
            df = pd.DataFrame(results)
            df[self.time_stamp_column] = pd.to_datetime(df[time_stamp_column])
            df = df.set_index(self.time_stamp_column, drop=True)
            hist_df = self.historical_data.set_index(self.time_stamp_column, drop=True)
            # Privilege new data in case of conflict -- may leave NaN in data
            df = df.join(hist_df, how='outer'), rsuffix ='_r').loc[:,df.columns]
            self.historical_data = df.reset_index()
    

    def derive_variables(self, now, independent_variable_values={}):
        day_of_week = now.weekday()
        hour_of_day = now.hour
        df = self.historical_data

        filter = (  (df[self.time_stamp_column].dt.weekday == day_of_week)
                  & (df[self.time_stamp_column].dt.hour >= hour_of_day - self.time_diff_tolerance)
                  & ((df[self.time_stamp_column].dt.hour < hour_of_day + self.time_diff_tolerance)
                    | ((df[self.time_stamp_column].dt.hour == hour_of_day + self.time_diff_tolerance) # for frequency >= 1min and < 1hr
                     & (df[self.time_stamp_column].dt.minute == 0))))
        if hour_of_day > 23-self.time_diff_tolerance:
            delta = (hour_of_day + self.time_diff_tolerance) % 24 # last hour to include in the next day
            filter |= (  (df[self.time_stamp_column].dt.weekday == (day_of_week + 1) % 7)
                    & (df[self.time_stamp_column].dt.hour < delta))
            filter |= (  (df[self.time_stamp_column].dt.weekday == (day_of_week + 1) % 7)
                    & (df[self.time_stamp_column].dt.hour == delta)
                    & (df[self.time_stamp_column].dt.minute == 0))
        if hour_of_day < 0 + self.time_diff_tolerance:
            delta = (hour_of_day - self.time_diff_tolerance) % 24  # first hour to include in the previous day
            filter |= (  (df[self.time_stamp_column].dt.weekday == (day_of_week - 1) % 7)
                    & (df[self.time_stamp_column].dt.hour >= delta))
        
        for variable, value in independent_variable_values.iteritems():
            tolerance = self.independent_variable_tolerances.get(variable, None)
            if tolerance is None:
                continue
            # Note: this will also filter any records where independent variable is NaN
            filter &= (df[variable] >= value-tolerance) & (df[variable] <= value+tolerance)

        filtered_data = df[filter]
        if filtered_data.shape[0] >= self.min_samples:
            if self.num_retries > 0 :
                self.adjust_tolerances(1/2**self.num_retries)
                self.num_retries = 0
            return self.build_results(filtered_data)
        else:
            if self.num_retries == self.max_retries:
                _log.debug("Maximum number of tolerance increases {} reached. "
                           "Defaulting to all-of-data.".format(self.max_retries))
                return self.build_results(df)
            _log.debug("Number of samples {} is less than the minimum {}. Increasing tolerances "
                       "by a factor of 2.".format(filtered_data.shape[0], self.min_samples))
            self.adjust_tolerances(2)
            self.num_retries += 1
            return self.derive_variables(now, independent_variable_values)
    

    def build_results(self, data):
        results = {}
        for dep in self.dependent_variables:
            # NaN values in dependent variables usually co-occur with NaNs in 
            # indpendent variables, so are already filtered. Filter them anyway.
            loc, (_, ub) = self.estimate_center(data[dep].loc[~pd.isnull(data[dep])])
            results[dep] = loc
            results[dep+"_ub"] = ub
        return results
    

    def adjust_tolerances(self, factor):
        self.time_diff_tolerance *= factor
        for k in self.independent_variable_tolerances:
            self.independent_variable_tolerances[k] *= factor
    

    def get_loc_scale(self, a, scale_method='MAD'):
        """Compute measure of central tendency and spread

        :param a: data
        :param scale_method: one of 'MAD' (median absolute distance) or 
            'MSE' (mean squared error) for maeasuring spread; the minimum
            of this metric is chosen as the center (median and mean, 
            respectively)
        :returns estimates of location and scale
        :rtype tuple(float)
        """
        if len(a) < 2:
            _log.warning("Not enough samples to estimate distribution parameters.")
            return a[0], 0
        if scale_method == 'MAD':
            _c = 0.6744897501960817  # st.norm.ppf(3/4)
            med = np.median(a)
            return med, np.median(np.absolute(med - a))/_c
        if scale_method == 'MSE':
            return np.mean(a), np.std(a)
        _log.debug("Location and scale estimation method {} not found; defaulting to MSE".format(scale_method))
        return get_loc_scale(a, 'MAD')


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
        :returns estimates of tolerance bounds
        :rtype tuple(float)
        """
        n = len(x)
        if n < 2:
            _log.warning("Not enough samples to estimate distribution parameters.")
            return x[0], (-np.inf, np.inf)
        df = n-1
        
        if loc is None:
            loc = x.mean()
        if scale is None:
            scale = x.std()
        
        z_p = st.norm.ppf(1-p, loc=loc, scale=scale)
        ncp = np.sqrt(n) * z_p
        t_a = st.nct.ppf(1-alpha, df, ncp)
        K = t_a/np.sqrt(n)

        return sorted((loc - scale*K, loc + scale*K))
            

    def estimate_center(self, data):
        """Estimate center and tolerance interval of 1D data
            
        :param data: load data
        :param scale_method: kwarg for get_loc_scale
        :param alpha: kwarg for tolerance_interval
        :param p: kwarg for tolerance_interval
        returns estimates of location and tolerance bounds
        tuple[float]: lower and upper bounds of confidence region
        """
        loc, scale = self.get_loc_scale(data, scale_method=self.scale_method)
        _data = (data - loc)/scale  # stabilize numerical errors
        lb, ub = self.tolerance_interval(_data, self.alpha, self.p, loc=0, scale=1)
        return loc, (lb*scale+loc, ub*scale+loc)

