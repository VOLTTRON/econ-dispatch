import numpy as np
import random
import csv
import re


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
            optimization_keys.sort(key=natural_keys)
            forecast_keys = flat_forecasts.keys()
            forecast_keys.sort(key=natural_keys)
            self.csv_file = csv.DictWriter(self.file, ["timestamp", "Optimization Status"] + optimization_keys + forecast_keys, extrasaction='ignore')
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
