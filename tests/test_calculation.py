import pytest
from numpy import ndarray, random
from sys import path

path.append(".")
from cvxEDA import cvxEDA


@pytest.mark.parametrize("n_array", [100, 1000, 10000, 1000000])
@pytest.mark.parametrize("frequency", [2, 8, 16, 32, 64])
def test_random_eda(n_array: int, frequency: int):
    y: ndarray = random.rand(n_array)
    yn: ndarray = (y - y.mean()) / y.std()
    results = cvxEDA(yn, 1.0 / frequency)
    assert all([isinstance(val, ndarray) for val in results.values()])
