from mock import patch, Mock
import numpy as np
import pytest

from anml.parameter.variables import Variable
from anml.parameter.parameter import ParameterSet
from anml.parameter.prior import GaussianPrior
from anml.solvers.base import ScipyOpt, ClosedFormSolver

from sfma.models import LinearMixedEffectsMarginal, GaussianRandomEffects, UModel


@pytest.fixture
def lme_inputs():
    np.random.seed(42)
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

    return mock_data, X, Z, beta_true, gamma_true

@patch.object(ParameterSet, '__init__', lambda x: None)
def test_lme_marginal(lme_inputs):
    data, X, Z, beta_true, gamma_true = lme_inputs
    n_beta, n_gamma = len(beta_true), len(gamma_true)
    # mock parameter set
    param_set = ParameterSet()
    param_set.reset()
    param_set.num_fe = n_beta
    param_set.num_re_var = n_gamma
    param_set.design_matrix = X
    param_set.design_matrix_re = Z
    param_set.re_var_diag_matrix = np.identity(n_gamma)
    param_set.constr_matrix_full = None
    param_set.lower_bounds_full = np.array([-10.0] * (n_beta + n_gamma))
    param_set.upper_bounds_full = np.array([10.0] * (n_beta + n_gamma))
    param_set.prior_fun = lambda x: 0.0
    
    model = LinearMixedEffectsMarginal(param_set)
    solver = ScipyOpt(model)
    x_init = np.random.rand(len(beta_true) + len(gamma_true))
    solver.fit(x_init, data, options=dict(solver_options=dict(maxiter=100)))
    np.testing.assert_allclose(solver.x_opt[:len(beta_true)], beta_true, rtol=5e-2)


@pytest.fixture
def gre_inputs():
    np.random.seed(42)
    n_groups, n_data_per_group, eta = 5, 50, 1.0
    Z = np.kron(np.identity(n_groups), np.ones((n_data_per_group, 1)))
    u_true = np.random.randn(n_groups) * eta 
    S = np.random.rand(n_groups * n_data_per_group)*0.01 + 0.01
    e = np.random.randn(n_groups * n_data_per_group) * S
    y = np.dot(Z, u_true) + e

    # mock data
    mock_data = Mock()
    mock_data.obs = y
    mock_data.obs_se = S
    
    return mock_data, Z, u_true, eta 


@patch.object(ParameterSet, '__init__', lambda x: None)
def test_gaussian_random_effects_model(gre_inputs):
    data, Z, u_true, eta = gre_inputs
    n_groups = len(u_true)
    param_set = ParameterSet()
    param_set.reset()
    with patch.object(ParameterSet, 'num_re', n_groups):
        param_set.num_fe = 1
        param_set.design_matrix_re = Z
        param_set.constr_matrix_full = None
        param_set.lower_bounds_full = [0.0] + [-2.0] * n_groups
        param_set.upper_bounds_full = [0.0] + [2.0] * n_groups
        param_set.re_priors = [GaussianPrior(mean=[0.0], std=[eta])]
        param_set.re_var_diag_matrix = np.ones((n_groups, 1))

        model = GaussianRandomEffects(param_set)
        solver = ScipyOpt(model)
        x_init = np.random.rand(n_groups)
        solver.fit(x_init, data, options=dict(solver_options=dict(maxiter=100)))
        assert np.linalg.norm(solver.x_opt - u_true) / np.linalg.norm(u_true) < 2e-2

        u_model = UModel(param_set)
        cf_solver = ClosedFormSolver(u_model)
        cf_solver.fit(x_init, data)
        assert np.linalg.norm(cf_solver.x_opt - u_true) / np.linalg.norm(u_true) < 2e-2

        


