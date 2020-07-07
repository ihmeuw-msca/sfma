from mock import patch, Mock, PropertyMock
import numpy as np
import pytest

from anml.parameter.variables import Variable
from anml.parameter.parameter import ParameterSet
from anml.parameter.prior import GaussianPrior, Prior
from anml.solvers.base import ScipyOpt, ClosedFormSolver, IPOPTSolver

from sfma.models.marginal import BetaGammaModel, BetaGammaSigmaModel, SimpleBetaGammaEtaModel
from sfma.models.maximal import UModel


@pytest.fixture
def lme_inputs():
    np.random.seed(16)
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


def test_beta_gamma_model(lme_inputs):
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


@pytest.fixture
def re_inputs():
    np.random.seed(16)
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


def test_u_model(re_inputs):
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


@pytest.fixture
def sfa_inputs():
    np.random.seed(11)
    n_data, n_beta = 200, 3
    beta_true = np.random.randn(n_beta)
    gamma_true = 0.1
    eta_true = 0.1
    X = np.random.randn(n_data, n_beta) * 2
    Z = np.identity(n_data)
    s_true = np.random.randn(n_data) * 0.05
    u = np.random.randn(n_data) * np.sqrt(gamma_true)
    v = np.maximum(0, np.random.randn(n_data) * np.sqrt(eta_true))
    y = X.dot(beta_true) + u - v

    # mock data 
    mock_data = Mock()
    mock_data.obs = y 
    mock_data.y = y 
    mock_data.obs_se = s_true
    mock_data.sigma2 = s_true**2

    return mock_data, X, Z, beta_true, gamma_true, eta_true


def test_beta_gamma_eta_model(sfa_inputs):
    data, X, Z, beta_true, gamma_true, eta_true = sfa_inputs
    n_beta = len(beta_true)
    with patch.object(ParameterSet, '__init__', lambda x: None):
        param_set = ParameterSet()
        param_set.reset()
        param_set.num_fe = n_beta
        param_set.num_re_var = 1
        param_set.design_matrix_fe = X
        param_set.design_matrix_re = Z
        param_set.lb_fe = [-10.0] * n_beta
        param_set.ub_fe = [10.0] * n_beta
        param_set.lb_re_var = [0.0] * 1
        param_set.ub_re_var = [10.0] * 1
        param_set.constr_matrix_fe = np.zeros((1, n_beta))
        param_set.constr_lb_fe = [0.0]
        param_set.constr_ub_fe = [0.0]
        param_set.constr_matrix_re_var = np.zeros((1, 1))
        param_set.constr_lb_re_var = [0.0]
        param_set.constr_ub_re_var = [0.0]
        param_set.fe_priors = []
        param_set.re_var_priors = []
        
        model = SimpleBetaGammaEtaModel(param_set)
        solver = IPOPTSolver(model)
        x_init = np.random.rand(len(beta_true) + 2)
        solver.fit(x_init, data, options=dict(solver_options=dict(max_iter=100)))
        assert np.linalg.norm(solver.x_opt[:n_beta] - beta_true) / np.linalg.norm(beta_true) < 2e-2
        

        


