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
""".. todo:: Module docstring"""
from io import StringIO
import csv
from datetime import timedelta
import json
import logging
import os.path
import random
import re

import pytz
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers


LOG = logging.getLogger(__name__)

def atoi(text):
    """cast string digits to int for sorting"""
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """cast each character of a string as `str` or `int` using `atoi`.

    For "human sorting", http://nedbatchelder.com/blog/200712/human_sorting.html.
    Found here: http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside#5967539

    :param text: string containing mixed letters and digits
    :returns: list of mixed str and int, for letters and diits, respectively
    :rtype: list
    """
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def records_fix(data):
    """Re-structure data formatted as [{key: value}]
    
    :param data: 
    :type data: list of dicts
    :returns: properly structured training data
    :rtype: dict of lists
    """
    keys = data[0].keys()
    with StringIO() as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(data)
        f.seek(0)
        results = csv_file_fix(f)

    return results

def csv_file_fix(file_obj):
    """Re-structure data from file or StringIO stream

    :param file_obj: first argument to pandas.read_csv
    :type file_obj: string, path, or file-like. See pandas documentation for details.
    :returns: properly structured training data
    :rtype: dict of lists
    """
    try:
        df = pd.read_csv(file_obj, header=0, parse_dates=["timestamp"])
    except ValueError:
        #Retry if we don't have a timestamp column.
        df = pd.read_csv(file_obj, header=0)
    results = {k: df[k].values for k in df}
    return results

def historian_data_fix(data):
    """Re-structure data formatted as {topic: (timestamp, value)}

    :param data: Volttron historian-formatted data
    :type data: dict of 2-tuples
    :returns: properly structured training data
    :rtype: dict of lists
    """
    results = {}
    for key, values in data.items():
        time_stamps = pd.to_datetime([x[0] for x in values]).floor("1min")
        readings = pd.Series((x[1] for x in values), index=time_stamps)

        results[key] = readings

    df = pd.DataFrame(results).dropna()

    results = {k: df[k].values for k in data}

    results["timestamp"] = df.index.values

    return results

def normalize_training_data(data):
    """Parse variously structured data into common structure

    Outputs data as equal-length dicts, organized by key.
    One key is always "timestamp".

    :param data: data to be normalized
    :returns: properly structured data
    :rtype: dict of lists
    """
    if not data:
        return {}

    if isinstance(data, list):
        # Assume list of dicts from CSV file
        return records_fix(data)

    if isinstance(data, str):
        # Assume file name
        if data.endswith("csv"):
            return csv_file_fix(data)

        if data.endswith("json"):
            with open(data, "rb") as f:
                return normalize_training_data(json.load(f))

    if isinstance(data, dict):
        values = data.get("values")
        if isinstance(values, dict):
            # Data returned from historian.
            return historian_data_fix(values)
        else:
            # Probably a json file from the config store.
            result = {k: np.array(v) for k, v in data.items()}
        return result

    return None

class OptimizerCSVOutput(object):
    """Handles I/O for optimizer debug output

    :param file_name: path to output file
    """
    def __init__(self, file_name):
        self.file_name = file_name
        try:
            with open(file_name, "wb"):
                pass
        except IOError as e:
            LOG.error("Problem opening file {}".format(file_name))
            raise e
        self.columns = None

    def writerow(self,
                 timestamp,
                 results,
                 forecasts={},
                 singleton_columns=['timestamp']):
        """Write a single row of debug output. Initialize if first row

        :param timestamp: beginning of optimization window
        :type timestamp: datetime.datetime
        :param results: results to write
        :param forecasts: forecasts from optimization
        :param singleton_columns: neither forecasts nor optimization variables
        """
        flat_forecasts = {}
        for i, record in enumerate(forecasts):
            for k, v in record.items():
                if k.lower() == 'timestamp':
                    continue
                flat_forecasts["{}_{}".format(k, i)] = v

        # initialize self.columns
        if self.columns is None:
            results_keys = results.keys()
            for k in singleton_columns:
                try:
                    results_keys.remove(k)
                except ValueError:
                    pass
            forecast_keys = flat_forecasts.keys()
            # sort keys so, e.g., key9 and key10 are next to each other
            results_keys.sort(key=natural_keys)
            forecast_keys.sort(key=natural_keys)
            self.columns = singleton_columns + results_keys + forecast_keys
            with open(self.file_name, 'ab') as f:
                csv_file = csv.DictWriter(f, self.columns, extrasaction='ignore')
                csv_file.writeheader()

        row = {}
        row["timestamp"] = str(timestamp)
        row.update(results)
        row.update(flat_forecasts)

        try:
            with open(self.file_name, 'ab') as f:
                csv_file = csv.DictWriter(f, self.columns, extrasaction='ignore')
                csv_file.writerow(row)
        except IOError:
            LOG.error("Failed to open {}. Would have written row: {}".format(self.file_name, row))


class PiecewiseError(Exception):
    """Inidcate failure of piecewise-linear curve fit"""
    pass


def clean_training_data(inputs,
                        outputs,
                        capacity,
                        timestamps=None,
                        min_cap_ratio=0.0,
                        min_time_coverage=365):
    """Test if training data meets standards, then clean it up

    :param inputs: dependent variable of curve fit
    :param outputs: independent variable of curve fit
    :param capacity: upper bound on interval of output to fit over
        (lower is assumed to be zero)
    :param timestamps: data timestamps to ensure representative sample
    :param min_cap_ratio: minimum acceptable ratio of max(data) to capacity
    :param min_time_coverage: minimum acceptable time range (in days)
    :returns: inputs, outputs -- cleaned version of parameters
    :rtype: tuple of numpy.array
    :raises ValueError: if training data does not meet standards
    """
    if (min(inputs) < 0) or (min(outputs) < 0):
        raise ValueError("Training data is not non-negative")

    if len(inputs) != len(outputs):
        raise ValueError("Training inputs and outputs are not the same size")

    max_x = max(outputs)
    cap_ratio = float(max_x)/capacity

    if cap_ratio > 1.0:
        raise ValueError("max(outputs) {} is greater than capacity {}"
                         "".format(max_x, capacity))
    if cap_ratio < min_cap_ratio:
        raise ValueError("Ratio of max(outputs)/capacity "
                         "({}/{}={}) below min_cap_ratio ({}).".format(
                             capacity, max_x, cap_ratio, min_cap_ratio))

    if timestamps is not None:
        time_start = min(timestamps)
        time_end = max(timestamps)
        time_delta = time_end - time_start
        if time_delta < timedelta(days=min_time_coverage):
            raise ValueError("Training data does not represent the minimum "
                             "time range {} days. Date range: {} to {}".format(
                                 min_time_coverage, time_start, time_end))

    # filter NaN
    inputs, outputs = pd.Series(inputs), pd.Series(outputs)
    valid = (~inputs.isnull() & ~outputs.isnull())
    inputs, outputs = inputs[valid].values, outputs[valid].values

    return inputs, outputs

def fit_prime_mover(
        x, y, xmin=None, xmax=None, n_fine=100, epsilon=1e-10, verbose=False):
    """Solves a quadratic program to fit the function
    `y = p0*x/(p1+p2*x+p3*x**2)` without singularities. Ensures
    `p1+p2*x+p3*x**3 >= epsilon` for all x in a fine mesh

    :param x: independent variable to fit
    :param y: dependent variable to fit
    :param xmin: lower limit of mesh
    :param xmax: upper limit of mesh
    :param n_fine: number of points in mesh
    :param epsilon: small value to enforce > 0
    :param verbose: verbosity of `cvxopt` solver
    """
    solvers.options['show_progress'] = verbose

    if xmin is None:
        xmin = 0

    if xmax is None:
        xmax = max(x)

    fine_x = np.linspace(xmin, xmax, n_fine)

    x2 = x ** 2
    x3 = x ** 3
    x4 = x ** 4
    y2 = y ** 2

    P = np.zeros((4, 4))
    P[0, 0] = sum(x2)
    P[0, 1] = sum(-x * y)
    P[1, 0] = sum(-x * y)
    P[0, 2] = sum(-x2 * y)
    P[2, 0] = sum(-x2 * y)
    P[0, 3] = sum(-x3 * y)
    P[3, 0] = sum(-x3 * y)
    P[1, 1] = sum(y2)
    P[1, 2] = sum(x * y2)
    P[2, 1] = sum(x * y2)
    P[1, 3] = sum(x2 * y2)
    P[3, 1] = sum(x2 * y2)
    P[2, 2] = sum(x2 * y2)
    P[2, 3] = sum(x3 * y2)
    P[3, 2] = sum(x3 * y2)
    P[3, 3] = sum(x4 * y2)
    P = 2 * P.T

    q = np.zeros((4, 1))

    if n_fine > 0:
        G = np.zeros((n_fine, 4))
        G[:, 1] = np.ones((1, n_fine))
        G[:, 2] = fine_x
        G[:, 3] = fine_x ** 2
        G = -1 * G

        h = -epsilon * np.ones((n_fine, 1))
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    else:
        sol = solvers.qp(matrix(P), matrix(q))

    return np.array(sol['x'])[:, 0]

def piecewise_linear(inputs,
                     outputs,
                     capacity,
                     segment_target=5,
                     curve_type='poly',
                     **kwargs):
    """Compute a piece-wise linear curve from inputs, outputs, and capacity

    :param inputs: dependent variable values
    :param outputs: independent variable values
    :param capacity: max output value regardless of recorded output values
    :param segment_target: number of segments to target. Failure to hit this
        target after 50 iterations is an error.
    :param curve_type: type of regression to perform: 'poly' for numpy
        polyfit, 'prime_mover' for cvxopt "prime mover" curve
    :param kwargs: keyword arguments to regression function
    :returns: piecewise-linear curve definition in the form (a, b, xmin, xmax)
    :rtype: 4-tuple of floats
    """
    x_values = outputs
    y_values = inputs
    max_x = capacity
    max_y = max(y_values)
    resolution = 100.0 # number of points in inter-/extrapolation
    error_threshold_max = 1.0
    error_threshold_min = 0.0
    error_threshold = 0.5

    LOG.debug("Max X: {}, max Y: {}".format(max_x, max_y))

    if curve_type == 'poly':
        # default is quadratic fit
        deg = kwargs.pop('deg', 2)
        params = np.polyfit(x_values, y_values, deg=deg, **kwargs)
        curve_func = lambda x, *params: np.polyval(params, x)
    elif curve_type == 'prime_mover':
        params = fit_prime_mover(x_values, y_values, **kwargs)
        curve_func = lambda x, P0, P1, P2, P3: P0*x/(P1 + P2*x + P3*x**2)
    else:
        raise ValueError("Unimplemented regression option '{}'. Choose from "
                         "'poly' or 'prime_mover'.".format(curve_type))

    LOG.debug("Curve type: {}. Regression Coefs: {}".format(curve_type, params))

    x0 = min(x_values)
    y0 = curve_func(x0, *params)
    x1 = x0 + (1.0 / resolution) * (1.0 - x0)
    y1 = curve_func(x1, *params)
    initial_coeff = np.polyfit([x0, x1], [y0, y1], 1)

    max_iterations = 50

    for _ in range(1, max_iterations+1):
        xmin = [x0]
        xmax = []
        coeffarray1 = initial_coeff[0]
        coeffarray2 = initial_coeff[1]

        for i in range(2, int(resolution)+1):
            xn = x0+(float(i)/resolution)*(max_x-x0)
            yn = curve_func(xn, *params)
            yp = coeffarray2 + coeffarray1 * xn
            error = abs(yn-yp)/max_y

            if error > error_threshold:
                xn_1 = x0+((i-1)/resolution)*(max_x-x0)
                yn_1 = curve_func(xn_1, *params)
                err_coeff = np.polyfit([xn_1, xn], [yn_1, yn], 1)
                coeffarray1 = err_coeff[0]
                coeffarray2 = err_coeff[1]
                xmin.append(xn_1)
                xmax.append(xn_1)

        xmax.append(xn)

        segment_total = len(xmin)
        old_error_threshold = error_threshold
        if segment_total < segment_target:
            error_threshold_max = error_threshold
        elif segment_total > segment_target:
            error_threshold_min = error_threshold
        else:
            break
        error_threshold = (error_threshold_max + error_threshold_min) / 2.0
        LOG.debug("Segments: {} Old Error Thresh: {} New Error Thresh: {}"
                  "".format(
                      segment_total, old_error_threshold, error_threshold))

    else:
        if segment_total < segment_target and error_threshold < 0.05:
            #add one segment if you are accurate, but need more segments
            for i in range(segment_target-segment_total):
                xmin.append((xmin[-1]+xmax[-1])/2)
                xmax_end = xmax[-1]
                xmax = xmax[:-1]
                xmax.append(xmin[-1])
                xmax.append(xmax_end)
            #break
        else:
            raise PiecewiseError("Max iterations hit while testing for target "
                                 "segment count.")

    a = []
    b = []
    for x1, x2 in zip(xmin, xmax):
        y1 = curve_func(x1, *params)
        y2 = curve_func(x2, *params)
        coeff = np.polyfit([x1, x2], [y1, y2], 1)
        a.append(coeff[0])
        b.append(coeff[1])

    return a, b, xmin, xmax

def _get_default_curve_data(curve_name):
    base_path = os.path.dirname(__file__)
    file_name = os.path.join(base_path, "component_models",
                             "normalized_default_curves",
                             "{}.csv".format(curve_name))
    if not os.path.exists(file_name):
        raise ValueError("Invalid default curve: {}".format(curve_name))

    data_frame = pd.read_csv(file_name)
    if "eff" in data_frame:
        eff_cop_div_rated = data_frame["eff"].values
    else:
        eff_cop_div_rated = data_frame["cop"].values

    plf = data_frame["plf"].values

    return eff_cop_div_rated, plf

# TODO Make this configurable
_conversion_factors = {"centrifugal_chiller_igv": 3.5168,
                       "absorption_chiller": 0.012,
                       "micro_turbine_generator":
                           0.8/3.28**3/1026/1055.06*1000}
_conversion_factors["fuel_cell"] = \
    _conversion_factors["micro_turbine_generator"]

def get_default_curve(curve_name, capacity, rated_eff_cop):
    """Retrieve default curve and adapt to component parameters

    :param curve_name: name component type (base name of the CSV file to load)
    :param capacity: capacity of the component
    :param rated_eff_cop: efficiency or COP (depends on the component what
        you call it but it's used the same either way...)
    :param conversion_factor: unit conversion factor
    :return: input and output values to be passed into the training function
    :rtype: dict
    """
    eff_cop_div_rated, plf = _get_default_curve_data(curve_name)

    conversion_factor = _conversion_factors.get(curve_name, 1.0)

    outputs = plf * capacity
    inputs = outputs * conversion_factor / (rated_eff_cop * eff_cop_div_rated)

    return {"outputs": outputs, "inputs": inputs}

def least_squares_regression(inputs=None, output=None):
    """Regress outputs on inputs using linear least squares with intercept"""
    if inputs is None:
        raise ValueError("At least one input column is required")
    if output is None:
        raise ValueError("Output column is required")

    if type(inputs) != tuple:
        inputs = (inputs,)

    ones = np.ones(len(inputs[0]))
    x_columns = np.column_stack((ones,) + inputs)

    solution, resid, rank, s = np.linalg.lstsq(x_columns, output)

    return solution

def preprocess(df,
               timezone={},
               renamings={},
               linspec={},
               nonlinspec={},
               bounds={},
               decision_variables=None):
    """ Pre-process data in a number of ways

        Performing the following operations in order:

        1. convert local timestamps to UTC
        2. rename variables
        3. form linear combinations of variables
        4. multiply variables together
        5. enforce lower (and possibly upper) bounds
        6. finally retain only relevant variables

        :param df: data to process
        :type df: pandas.dataframe
        :param timezone: dict mapping timestamp column names to pytz
            timezone name
        :param renamings: dict mapping new column names to old
        :param linspec: dict mapping new column names to lists of tuples
            holding variable names and their coefficient for linear
            combinations; add a constant using the name `__constant__`
        :param nonlinspec: dict mapping new column names to lists holding
            variables to be multiplied together
        :param bounds: dict of tuples mapping column names to lower bound
            and optional upper bound
        :param decision_variables: list of relevant variables

        :returns pre-processed data
        :rtype pandas.DataFrame
    """
    for k, v in timezone.items():
        tz = pytz.timezone(v)
        df[k] = df[k].apply(lambda ts: ts.replace(tzinfo=tz))
        df[k] = df[k].apply(lambda ts: ts.astimezone(pytz.UTC))
    for k, v in renamings.items():
        df[k] = df[v]
    for k, v in linspec.items():
        _data = pd.DataFrame()
        for vv, m in v:
            if vv == "__constant__":
                _data[vv] = np.ones(df.shape[0])*m
            else:
                _data[vv] = df[vv]*m
        df[k] = _data.sum(axis=1, skipna=False)
    for k, v in nonlinspec.items():
        df[k] = df[v].prod(axis=1, skipna=False)
    for k, v in bounds.items():
        # assert (len(v) == 1) or (len(v) == 2)
        _data = df[k].copy()
        _test = _data >= v[0]
        if len(v) == 2:
            _test &= _data < v[1]
        _data[np.where(~_test)[0]] = np.nan
        df[k] = _data
    if decision_variables is not None:
        df = df[decision_variables]
    return df

def round_to_hour(dt):
    """Round a datetime to the nearest hour

    :param dt: timestamp
    :type dt: datetime.datetime
    :returns: rounded timestamp
    """
    dt_start_of_hour = dt.replace(minute=0, second=0, microsecond=0)
    dt_half_hour = dt.replace(minute=30, second=0, microsecond=0)

    if dt >= dt_half_hour:
        # round up
        dt = dt_start_of_hour + timedelta(hours=1)
    else:
        # round down
        dt = dt_start_of_hour

    return dt
