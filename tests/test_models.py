from mock import patch, Mock, PropertyMock
import numpy as np
import pytest

from anml.parameter.variables import Variable
from anml.parameter.parameter import ParameterSet
from anml.parameter.prior import GaussianPrior, Prior
from anml.solvers.base import ScipyOpt, ClosedFormSolver, IPOPTSolver

from sfma.models.marginal import BetaGammaModel, BetaGammaSigmaModel
from sfma.models.maximal import UModel


@pytest.fixture
def lme_inputs():
    np.random.seed(42)
    n_data, n_beta, n_gamma = 50, 3, 2
    beta_true = np.random.randn(n_beta)
    gamma_true = np.random.rand(n_gamma)*0.09 + 0.01
    X = np.random.randn(n_data, n_beta)
    Z = np.random.randn(n_data, n_gamma)
    s_true = 0.05
    V = s_true**2 * np.ones(n_data)
    D = np.diag(V) + (Z*gamma_true).dot(Z.T)
    u = np.random.multivariate_normal(np.zeros(n_data), D)
    y = X.dot(beta_true) + u

    # mock data 
    mock_data = Mock()
    mock_data.obs = y 
    mock_data.y = y 
    mock_data.obs_se = s_true * np.ones(n_data)
    mock_data.sigma2 = s_true**2 * np.ones(n_data)

    return mock_data, X, Z, beta_true, gamma_true, s_true


def test_lme_marginal(lme_inputs):
    data, X, Z, beta_true, gamma_true, s_true = lme_inputs
    n_beta, n_gamma = len(beta_true), len(gamma_true)
    # mock parameter set
    with patch.object(ParameterSet, '__init__', lambda x: None):
        param_set = ParameterSet()
        param_set.reset()
        param_set.num_fe = n_beta
        param_set.num_re_var = n_gamma
        param_set.design_matrix_fe = X
        param_set.design_matrix_re = Z
        param_set.re_var_padding = np.identity(n_gamma)
        param_set.lb_fe = [-10.0] * n_beta
        param_set.ub_fe = [10.0] * n_beta
        param_set.lb_re_var = [0.0] * n_gamma
        param_set.ub_re_var = [10.0] * n_gamma
        param_set.constr_matrix_fe = np.zeros((1, n_beta))
        param_set.constr_lb_fe = [0.0]
        param_set.constr_ub_fe = [0.0]
        param_set.constr_matrix_re_var = np.zeros((1, n_gamma))
        param_set.constr_lb_re_var = [0.0]
        param_set.constr_ub_re_var = [0.0]
        param_set.fe_priors = []
        param_set.re_var_priors = []
        
        model = BetaGammaModel(param_set)
        solver = ScipyOpt(model)
        x_init = np.random.rand(len(beta_true) + len(gamma_true))
        solver.fit(x_init, data, options=dict(solver_options=dict(maxiter=100)))
        assert np.linalg.norm(solver.x_opt[:len(beta_true)] - beta_true) / np.linalg.norm(beta_true) < 2e-2

        model2 = BetaGammaSigmaModel(param_set, sigma2_prior=Prior(lower_bound=[0.0], upper_bound=[10.]))
        solver2 = IPOPTSolver(model2)
        x_init2 = np.random.rand(len(beta_true) + len(gamma_true) + 1)
        solver2.fit(x_init2, data, options=dict(solver_options=dict(maxiter=200)))
        assert np.linalg.norm(solver2.x_opt[:len(beta_true)] - beta_true) / np.linalg.norm(beta_true) < 2e-2


@pytest.fixture
def re_inputs():
    np.random.seed(42)
    n_groups, n_data_per_group, eta = 5, 50, 1.0
    Z = np.kron(np.identity(n_groups), np.ones((n_data_per_group, 1)))
    u_true = np.random.randn(n_groups) * eta 
    s = np.random.rand(n_groups * n_data_per_group)*0.01 + 0.01
    e = np.random.randn(n_groups * n_data_per_group) * s
    y = np.dot(Z, u_true) + e

    # mock data
    mock_data = Mock()
    mock_data.obs = y
    mock_data.y = y
    mock_data.obs_se = s
    mock_data.sigma2 = s**2
    return mock_data, Z, u_true, eta 


def test_random_effects_model(re_inputs):
    data, Z, u_true, eta = re_inputs
    n_groups = len(u_true)
    with patch.object(ParameterSet, '__init__', lambda x: None):
        with patch.object(ParameterSet, 'num_re', new_callable=PropertyMock) as mock_num_re:
            param_set = ParameterSet()
            param_set.reset()
            param_set.num_fe = 1
            mock_num_re.return_value = n_groups
            param_set.design_matrix_re = Z
            param_set.constr_matrix_re = np.zeros((1, n_groups))
            param_set.constr_lb_re = [0.0]
            param_set.constr_ub_re = [0.0]
            param_set.lb_re = [-2.0] * n_groups
            param_set.ub_re = [2.0] * n_groups
            param_set.re_priors = [GaussianPrior(mean=[0.0], std=[eta])]
            param_set.re_var_padding = np.ones((n_groups, 1))

            model = UModel(param_set)
            solver = ScipyOpt(model)
            x_init = np.random.rand(n_groups)
            solver.fit(x_init, data, options=dict(solver_options=dict(maxiter=100)))
            assert np.linalg.norm(solver.x_opt - u_true) / np.linalg.norm(u_true) < 2e-2

            u_model = UModel(param_set)
            cf_solver = ClosedFormSolver(u_model)
            cf_solver.fit(x_init, data)
            assert np.linalg.norm(cf_solver.x_opt - u_true) / np.linalg.norm(u_true) < 2e-2

        


