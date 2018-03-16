import numpy as np
import pandas as pd
import random
import csv
import re
import simplejson as json
from cStringIO import StringIO


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


if __name__ == '__main__':
    _test_regression()
