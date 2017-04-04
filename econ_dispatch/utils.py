import numpy as np
import random


def least_squares_regression(y, *columns):
    ones = np.ones(len(columns[0]))
    columns = (ones,) + columns
    
    cols = np.column_stack(columns)
    solution, resid, rank, s = np.linalg.lstsq(cols, y)
    return solution


def _test_regression():
    def one_input(x):
        return 2 + 3*x

    xs = range(100)
    ys = map(one_input, xs)

    solution = least_squares_regression(ys, xs)
    check = np.array([2.0, 3.0])
    assert np.allclose(solution, check)
    
    def two_inputs(x, y):
        return 2 + 3*x + 5*y

    xs = [random.randint(0,100) for _ in range(100)]
    ys = [random.randint(0,100) for _ in range(100)]
    zs = map(two_inputs, xs, ys)

    solution = least_squares_regression(zs, xs, ys)
    check = [2, 3, 5]
    assert np.allclose(solution, check)


if __name__ == '__main__':
    _test_regression()
