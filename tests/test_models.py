from mock import patch, Mock
import numpy as np
import pytest

from anml.parameter.variables import Variable
from anml.parameter.parameter import ParameterSet
from anml.solvers.base import ScipyOpt

from sfma.models import LinearMixedEffectsMarginal


@pytest.fixture
def inputs():
    n_data, n_beta, n_gamma = 50, 3, 2
    beta_true = np.random.randn(n_beta)
    gamma_true = np.random.rand(n_gamma)*0.09 + 0.01
    X = np.random.randn(n_data, n_beta)
    Z = np.random.randn(n_data, n_gamma)
    S = np.random.rand(n_data)*0.09 + 0.01
    V = S**2
    D = np.diag(V) + (Z*gamma_true).dot(Z.T)
    U = np.random.multivariate_normal(np.zeros(n_data), D)
    Y = X.dot(beta_true) + U 

    # mock data 
    mock_data = Mock()
    mock_data.obs = Y 
    mock_data.obs_se = S

    # mock parameter set
    mock_param_set = Mock()
    mock_param_set.num_fe = n_beta
    mock_param_set.num_re_var = n_gamma
    mock_param_set.design_matrix = X
    mock_param_set.design_matrix_re = Z
    mock_param_set.constr_matrix_full = np.identity(n_beta + n_gamma)
    mock_param_set.constr_lower_bounds_full = np.array([-10.0] * (n_beta + n_gamma))
    mock_param_set.constr_upper_bounds_full = np.array([10.0] * (n_data + n_gamma))
    mock_param_set.prior_fun = lambda x: 0.0
    return mock_data, mock_param_set, beta_true, gamma_true


def test_lme_marginal(inputs):
    data, param_set, beta_true, gamma_true = inputs
    model = LinearMixedEffectsMarginal(param_set)
    solver = ScipyOpt(model)
    x_init = np.random.rand(len(beta_true) + len(gamma_true))
    solver.fit(x_init, data, options=dict(maxiter=100))
    np.testing.assert_allclose(solver.x_opt[:len(beta_true)], beta_true, rtol=1e-2)











