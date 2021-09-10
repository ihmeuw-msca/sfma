"""
Test solver module
"""
import pytest
import numpy as np
from scipy.optimize import LinearConstraint

from sfma.solver import IPSolver, proj_csimplex


@pytest.fixture
def size():
    return 100


@pytest.fixture
def y(size):
    return np.random.randn(size)*5.0


@pytest.fixture
def hess():
    def hess(x):
        return np.identity(x.size)
    return hess


@pytest.fixture
def grad(y):
    def grad(x):
        return x - y
    return grad


@pytest.fixture
def linear_constraints(size):
    mat = np.identity(size)
    lb = -np.ones(size)
    ub = np.ones(size)
    return LinearConstraint(A=mat, lb=lb, ub=ub)


def test_solve_with_constraints(y, hess, grad, linear_constraints):
    solver = IPSolver(grad, hess, linear_constraints)
    p = solver.minimize(x0=np.zeros(y.size))
    assert np.allclose(p, np.maximum(-1.0, np.minimum(1.0, y)))


@pytest.mark.parametrize("w", [np.ones(10)])
@pytest.mark.parametrize("h", [9, 8, 7])
def test_proj_csimplex(w, h):
    w = proj_csimplex(w, h)
    assert np.allclose(w, h/w.size)
