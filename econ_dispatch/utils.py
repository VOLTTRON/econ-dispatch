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

import numpy as np
import pandas as pd
import random
import csv
import re
import json
import os.path
from cStringIO import StringIO
from datetime import timedelta
from scipy.optimize import curve_fit

import logging

_log = logging.getLogger(__name__)


def least_squares_regression(inputs=None, output=None):
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

def atoi(text):
    return int(text) if text.isdigit() else text

def records_fix(data):
    keys = data[0].keys()
    with StringIO() as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(data)
        f.seek(0)
        results = csv_file_fix(f)

    return results

def csv_file_fix(file_obj):
    try:
        df = pd.read_csv(file_obj, header=0, parse_dates=["timestamp"])
    except ValueError:
        #Retry if we don't have a timestamp column.
        df = pd.read_csv(file_obj, header=0)
    results = {k: df[k].values for k in df}
    return results


def historian_data_fix(data):
    results = {}
    for key, values in data.iteritems():
        time_stamps = pd.to_datetime([x[0] for x in values]).floor("1min")
        readings = pd.Series((x[1] for x in values), index=time_stamps)

        results[key] = readings

    df = pd.DataFrame(results).dropna()

    results = {k: df[k].values for k in data}

    results["timestamp"] = df.index.values

    return results
        

def normalize_training_data(data):
    if not data:
        return {}

    if isinstance(data, list):
        # Assume list of dicts from CSV file
        return records_fix(data)

    if isinstance(data,basestring):
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
            result = {k: np.array(v) for k,v in data.iteritems()}
        return result

    return None

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html

    Found here: http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside#5967539
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class OptimizerCSVOutput(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = open(file_name, "wb")
        self.csv_file = None

    def writerow(self, optimization, forecasts, timestamp=""):

        flat_forecasts = {}
        for i, record in enumerate(forecasts):
            flat_forecasts.update({"elec_load" + str(i):record.get("elec_load", 0.0),
                                   "heat_load" + str(i): record.get("heat_load", 0.0),
                                   "cool_load" + str(i): record.get("cool_load", 0.0),
                                   "solar_kW" + str(i): record.get("solar_kW", 0.0),
                                   "natural_gas_cost" + str(i): record.get("natural_gas_cost", 0.0),
                                   "electricity_cost" + str(i): record.get("electricity_cost", 0.0)
                                   })

        if self.csv_file is None:
            optimization_keys = optimization.keys()
            optimization_keys.remove("Optimization Status")
            optimization_keys.remove("Convergence Time")
            optimization_keys.remove("Objective Value")
            optimization_keys.sort(key=natural_keys)
            forecast_keys = flat_forecasts.keys()
            forecast_keys.sort(key=natural_keys)
            self.csv_file = csv.DictWriter(self.file,
                                           ["timestamp", "Optimization Status", "Objective Value", "Convergence Time"] +
                                           optimization_keys + forecast_keys, extrasaction='ignore')
            self.csv_file.writeheader()

        row = {}
        row["timestamp"] = str(timestamp)
        row.update(optimization)
        row.update(flat_forecasts)

        self.csv_file.writerow(row)

    def close(self):
        self.file.close()

def _test_regression():
    def one_input(x):
        return 2 + 3*x

    xs = range(100)
    ys = map(one_input, xs)

    solution = least_squares_regression(inputs=xs, output=ys)
    check = np.array([2.0, 3.0])
    assert np.allclose(solution, check)

    def two_inputs(x, y):
        return 2 + 3*x + 5*y

    a = [random.randint(0,100) for _ in range(100)]
    b = [random.randint(0,100) for _ in range(100)]
    c = map(two_inputs, a, b)

    solution = least_squares_regression(inputs=(a, b), output=c)
    check = [2, 3, 5]
    assert np.allclose(solution, check)

class PiecewiseError(StandardError):
    pass


def clean_training_data(inputs, outputs, capacity, timestamps=None, delete_outliers_sigmas=None, min_cap_ratio=0.9):
    """
    Test if training data meets common-sense tests and coverage standards, then clean it up a little.

    :param inputs:
    :param outputs:
    :param capacity:
    :param timestamps:
    :param delete_outliers_sigmas: TODO. Number of standard deviations from mean to define outliers.
    :param min_cap_ratio: how close to capacity to we require data
    :return inputs, outputs: cleaned version of parameters
    :raises ValueError: if training data does not meet standards
    """
    x_values = outputs
    y_values = inputs

    if (min(y_values) < 0) or (min(x_values) < 0):
        raise ValueError("Training data is not non-negative.")

    if len(y_values) != len(x_values):
        raise ValueError("Training data is not one-to-one.")

    max_x_value = max(x_values)
    cap_ratio = float(max_x_value)/float(max_x)

    if cap_ratio > 1.0:
        raise ValueError("max(outputs) {} is greater than capacity {}.".format(max_x_value,
                                                                               capacity))
    if cap_ratio < min_cap_ratio:
        raise ValueError("Ratio of max(outputs)/capacity "
                         "({}/{}={}) below min_cap_ratio ({}).".format(capacity,
                                                                       max_x_value,
                                                                       cap_ratio,
                                                                       min_cap_ratio))

    if timestamps is not None:
        time_start = min(timestamps)
        time_end = max(timestamps)
        time_delta = time_end - time_start
        if time_delta < timedelta(days=365):
            raise ValueError("Training data does not represent a full year of operation. "
                             "Date range: {} to {}".format(time_start,
                                                           time_end))

    valid = y_values > 0.0
    y_values = y_values[valid]
    x_values = x_values[valid]

    if delete_outliers_sigmas is not None:
        # what does this mean? noise should be around underlying curve
        pass

    return y_values, x_values


def piecewise_linear(inputs, outputs, capacity,
                     segment_target=5, curve_func=lambda x, a, b, c: a+b*x+c*x**2):
    """
    Produces a piecewise linear curve from the component inputs, outputs, and max capacity.

    inputs - input values for component
    outputs - output values for component
    capacity - sets the max output value regardless of the output values
    segment_target - Number of segments to target, failure to hit this target after 50
                    iterations is an error.
    regression_order - number of coefficients to use for the polyfit.
    min_cap_ratio - minimum ratio of max(outputs)/capacity allowed.
                    failure to exceed this value is an automatic failure.
    """
    x_values = outputs
    y_values = inputs
    max_x = capacity
    max_y = max(y_values)
    resolution = 100.0
    error_threshold_max = 1.0
    error_threshold_min = 0.0
    error_threshold = 0.5

    _log.debug("Max X: {}, max Y: {}".format(max_x, max_y))

    params, _ = curve_fit(curve_func, x_values, y_values)
    find_y = curve_func(x_values, *params)

    _log.debug("Regression Coefs: {}".format(params))

    x0 = min(x_values)
    y0 = find_y(x0)
    x1 = x0 + (1.0 / resolution) * (1.0 - x0)
    y1 = find_y(x1)
    initial_coeff = np.polyfit([x0, x1], [y0, y1], 1)

    max_iterations = 50

    for iteration in xrange(1, max_iterations+1):
        xmin = [x0]
        xmax = []
        coeffarray1 = initial_coeff[0]
        coeffarray2 = initial_coeff[1]

        for i in range(2, int(resolution)+1):
            xn = x0+(float(i)/resolution)*(max_x-x0)
            yn = find_y(xn)
            yp = coeffarray2 + coeffarray1 * xn
            error = abs(yn-yp)/max_y

            if error > error_threshold:
                xn_1 = x0+((i-1)/resolution)*(max_x-x0)
                yn_1 = find_y(xn_1)
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
        _log.debug("Segments: {} Old Error Thresh: {} New Error Thresh: {}".format(segment_total, old_error_threshold, error_threshold))

    else:
        raise PiecewiseError("Max iterations hit while testing for target segment count.")

    a = []
    b = []
    for x1, x2 in zip(xmin, xmax):
        y1 = find_y(x1)
        y2 = find_y(x2)
        coeff = np.polyfit([x1, x2], [y1, y2], 1)
        a.append(coeff[0])
        b.append(coeff[1])

    return a, b, xmin, xmax

def _test_piecewise():
    data = pd.read_csv("../component_test_data/test_piecewise_linear_chiller.csv")
    x = data["X"]
    y = data["Y"]

    valid = x > 0.0

    x = x[valid]
    y = y[valid]

    a, b, xmin, xmax = piecewise_linear(y.values, x.values, 600)

    print "a", a
    print "b", b
    print "xmin", xmin
    print "xmax", xmax


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

_conversion_factors = {"centrifugal_chiller_igv": 3.5168,
                       "absorption_chiller": 0.012,
                       "micro_turbine_generator": 0.8/3.28**3/1026/1055.06*1000}

_conversion_factors["fuel_cell"] =_conversion_factors["micro_turbine_generator"]

def get_default_curve(curve_name, capacity, rated_eff_cop):
    """
    :param curve_name: name component type (base name of the CSV file to load)
    :param capacity: capacity of the component
    :param rated_eff_cop: efficiency or COP (depends on the component what you call it but it's used the same either way...)
    :param conversion_factor: Factor to multiply the results by to convert units.
    :return: input values, output values ready to be passed into the training function
    """
    eff_cop_div_rated, plf = _get_default_curve_data(curve_name)

    conversion_factor = _conversion_factors.get(curve_name, 1.0)

    outputs = plf * capacity
    inputs = outputs * conversion_factor / (rated_eff_cop * eff_cop_div_rated)

    return {"outputs": outputs, "inputs": inputs}


def _test_default_curve():
    curve_tests = [["centrifugal_chiller_igv", 600, 5.5],
                   ["absorption_chiller", 150, 0.8],
                   ["boiler", 300, 0.9],
                   ["micro_turbine_generator", 300, 0.35]]

    for test in curve_tests:
        results = get_default_curve(*test)
        print test
        print "output", results["outputs"][:10]
        print "input", results["inputs"][:10]

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    _test_default_curve()
    _test_regression()
    _test_piecewise()
