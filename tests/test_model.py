"""
Test the model module
"""
import numpy as np
import pandas as pd
import pytest
from sfma import Data, GaussianPrior, UniformPrior, Variable
from sfma.model import SFMAModel


def ad_jacobian(fun, x, out_shape=(), eps=1e-10):
    c = x + 0j
    if np.isscalar(x):
        g = fun(x + eps*1j).imag/eps
    else:
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
def df():
    np.random.seed(123)
    return pd.DataFrame({
        "obs": np.random.randn(10),
        "obs_se": 0.2 + np.random.rand(10),
        "var": np.random.randn(10),
        "intercept": 1.0,
    })


@pytest.fixture
def data():
    return Data(obs="obs", obs_se="obs_se")


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


def test_mat(data, variables, df):
    model = SFMAModel(data, variables, df=df)
    assert np.allclose(model.mat, df[["intercept", "var"]].values)


def test_gprior(data, variables, df):
    model = SFMAModel(data, variables, df=df)
    linear_gvec = model.parameter.prior_dict["direct"]["GaussianPrior"].params
    assert np.allclose(linear_gvec,
                       np.array([[0.0, 0.0], [np.inf, 1.0]]))


def test_uprior(data, variables, df):
    model = SFMAModel(data, variables, True, True, df=df)
    assert np.allclose(model.cvec, np.array([0.0, 1.0]))


@pytest.mark.parametrize("beta", [np.arange(2)*1.0, np.ones(2)])
def test_gradient_beta(data, variables, df, beta):
    model = SFMAModel(data, variables, True, True, df=df)
    my_gradient = model.gradient_beta(beta)
    tr_gradient = ad_jacobian(model.objective_beta, beta)
    assert np.allclose(my_gradient, tr_gradient)


@pytest.mark.parametrize("beta", [np.arange(2)*1.0, np.ones(2)])
def test_hessian_beta(data, variables, df, beta):
    model = SFMAModel(data, variables, True, True, df=df)
    my_hess = model.hessian_beta(beta)
    tr_hess = ad_jacobian(model.gradient_beta, beta, out_shape=(beta.size,))
    assert np.allclose(my_hess, tr_hess)


@pytest.mark.parametrize("eta", [0.1, 1.0])
def test_gradient_eta(data, variables, df, eta):
    model = SFMAModel(data, variables, True, True, df=df)
    my_gradient = model.gradient_eta(eta)
    tr_gradient = ad_jacobian(model.objective_eta, eta)
    assert np.isclose(my_gradient, tr_gradient)


@pytest.mark.parametrize("gamma", [0.1, 1.0])
def test_gradient_gamma(data, variables, df, gamma):
    model = SFMAModel(data, variables, True, True, df=df)
    my_gradient = model.gradient_gamma(gamma)
    tr_gradient = ad_jacobian(model.objective_gamma, gamma)
    assert np.isclose(my_gradient, tr_gradient)
