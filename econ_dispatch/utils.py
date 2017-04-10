import numpy as np
import random


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
