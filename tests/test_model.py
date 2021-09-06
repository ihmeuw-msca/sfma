"""
Test the model module
"""
import pytest
import numpy as np
import pandas as pd
from regmod.prior import UniformPrior

from sfma.model import SFMAModel
from sfma import Data, Variable, GaussianPrior


def ad_jacobian(fun, x, out_shape=(), eps=1e-10):
    c = x + 0j
    g = np.zeros((*out_shape, *x.shape))
    if len(out_shape) == 0:
        for i in np.ndindex(x.shape):
            c[i] += eps*1j
            g[i] = fun(c).imag/eps
            c[i] -= eps*1j
    else:
        for j in np.ndindex(out_shape):
            for i in np.ndindex(x.shape):
                c[i] += eps*1j
                g[j][i] = fun(c)[j].imag/eps
                c[i] -= eps*1j
    return g


@pytest.fixture
def data():
    np.random.seed(123)
    df = pd.DataFrame({
        "obs": np.random.randn(10),
        "weights": 0.2 + np.random.rand(10),
        "var": np.random.randn(10)
    })
    return Data(col_obs="obs",
                col_covs=["var"],
                col_weights="weights",
                df=df)


@pytest.fixture
def gprior():
    return GaussianPrior(mean=0.0, sd=1.0)


@pytest.fixture
def uprior():
    return UniformPrior(lb=0.0, ub=1.0)


@pytest.fixture
def variables(gprior, uprior):
    return [
        Variable("intercept"),
        Variable("var", priors=[gprior, uprior])
    ]


def test_mat(data, variables):
    model = SFMAModel(data, variables)
    assert np.allclose(model.mat, data.df[["intercept", "var"]].values)


def test_gprior(data, variables):
    model = SFMAModel(data, variables)
    assert np.allclose(model.gvec,
                       np.array([[0.0, 0.0], [np.inf, 1.0]]))


@pytest.mark.parametrize("include_ie", [True, False])
@pytest.mark.parametrize("include_re", [True, False])
def test_uprior(data, variables, include_ie, include_re):
    model = SFMAModel(data, variables, include_ie, include_re)
    ub_ie = np.inf if include_ie else 0.0
    ub_re = np.inf if include_re else 0.0
    assert np.allclose(model.uvec,
                       np.array([[-np.inf, 0.0, 0.0, 0.0],
                                 [np.inf, 1.0, ub_ie, ub_re]]))


@pytest.mark.parametrize("x", [np.arange(4)*1.0, np.ones(4)])
def test_gradient(data, variables, x):
    model = SFMAModel(data, variables, True, True)
    my_gradient = model.gradient(x)
    tr_gradient = ad_jacobian(model.objective, x)
    assert np.allclose(my_gradient, tr_gradient)


@pytest.mark.parametrize("x", [np.arange(4)*1.0, np.ones(4)])
def test_hessian(data, variables, x):
    model = SFMAModel(data, variables, True, True)
    my_hess = model.hessian(x)
    tr_hess = ad_jacobian(model.gradient, x, out_shape=(x.size,))
    assert np.allclose(my_hess, tr_hess)
