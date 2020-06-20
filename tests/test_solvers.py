from mock import Mock, patch, PropertyMock
import numpy as np
import pytest

from anml.parameter.parameter import ParameterSet
from anml.parameter.prior import GaussianPrior, Prior

from sfma.models import LinearMixedEffectsMarginal, UModel, VModel
from sfma.solvers import AlternatingSolver


@pytest.fixture
def sfa_inputs():
    np.random.seed(42)
    n_data, n_beta, n_u, n_v = 200, 3, 2, 5
    gamma_true, eta_true = 1.0, 1.0
    beta_true = np.random.randn(n_beta)
    u_true = np.random.randn(n_u) * gamma_true
    v_true = np.maximum(0, np.random.randn(n_v) * eta_true)
    X = np.random.randn(n_data, n_beta)
    Z_u = np.random.randn(n_data, n_u)
    Z_v = np.kron(np.identity(n_v), np.ones((n_data // n_v, 1)))
    sigma = np.random.rand(n_data)*0.09 + 0.01
    e = np.random.randn(n_data) * sigma
    y = np.dot(X, beta_true) + np.dot(Z_u, u_true) - np.dot(Z_v, v_true) + e

    # mock data 
    mock_data = Mock()
    mock_data.obs = y 
    mock_data.input_obs = y
    mock_data.obs_se = sigma

    return mock_data, X, Z_u, Z_v, beta_true, gamma_true, eta_true, u_true, v_true


@patch.object(ParameterSet, '__init__', lambda x: None)
def test_alternating_solver(sfa_inputs):
    data, X, Z_u, Z_v, beta_true, gamma_true, eta_true, u_true, v_true = sfa_inputs
    # mock parameter set for beta and gammas
    n_beta = len(beta_true)
    param_set_beta_gamma = ParameterSet()
    param_set_beta_gamma.reset()
    param_set_beta_gamma.num_fe = n_beta
    param_set_beta_gamma.num_re_var = 1
    param_set_beta_gamma.design_matrix = X
    param_set_beta_gamma.design_matrix_re = Z_u
    param_set_beta_gamma.re_var_diag_matrix = np.ones((len(u_true), 1))
    param_set_beta_gamma.constr_matrix_full = None
    param_set_beta_gamma.lower_bounds_full = np.array([-10.0] * n_beta + [0.0])
    param_set_beta_gamma.upper_bounds_full = np.array([10.0] * (n_beta + 1))
    param_set_beta_gamma.prior_fun = lambda x: 0.0

    # mock parameter set for us
    param_set_u = ParameterSet()
    param_set_u.reset()
    with patch.object(ParameterSet, 'num_re', lambda x: x):
        param_set_u.num_fe = 1
        param_set_u.num_re = len(u_true)
        param_set_u.design_matrix_re = Z_u
        param_set_u.constr_matrix_full = None
        param_set_u.lower_bounds_full = [0.0] + [-2.0] * len(u_true)
        param_set_u.upper_bounds_full = [0.0] + [2.0] * len(u_true)
        param_set_u.re_priors = [GaussianPrior(mean=[0.0], std=[gamma_true])]
        param_set_u.re_var_diag_matrix = np.ones((len(u_true), 1))

        # mock parameter set for vs
        param_set_v = ParameterSet()
        param_set_v.reset()
        param_set_v.num_fe = 1
        param_set_v.num_re = len(v_true)
        param_set_v.design_matrix_re = Z_v
        param_set_v.constr_matrix_full = None
        param_set_v.lower_bounds_full = [0.0] + [-2.0] * len(v_true)
        param_set_v.upper_bounds_full = [0.0] + [2.0] * len(v_true)
        param_set_v.re_priors = [GaussianPrior(mean=[0.0], std=[eta_true], lower_bound=[0.0], upper_bound=[np.inf])]
        param_set_v.re_var_diag_matrix = np.ones((len(v_true), 1))

        data.params = [param_set_beta_gamma, param_set_u, param_set_v]
        alt_solver = AlternatingSolver()

        alt_solver.set_params(data)
        # at true us, vs, eta
        beta_init = np.zeros(len(beta_true))
        gamma_init = [np.random.rand()]

        betas, _, us, vs, _ = alt_solver.step(beta_init, gamma_init, u_true, v_true, eta_true, data)
        assert np.linalg.norm(betas - beta_true)/ np.linalg.norm(beta_true) < 2e-2
        assert np.linalg.norm(us - u_true)/ np.linalg.norm(u_true) < 2e-2
        assert np.linalg.norm(vs - v_true)/ np.linalg.norm(v_true) < 2e-2

        # at true beta, gamma, vs
        u_init = np.zeros(len(u_true))
        _, _, us, _, _ = alt_solver.step(beta_true, gamma_true, u_init, v_true, eta_true, data)
        assert np.linalg.norm(us - u_true)/ np.linalg.norm(u_true) < 2e-2

        # at true beta, gamma, us
        v_init = np.zeros(len(v_true)) + 0.1
        _, _, _, vs, _ = alt_solver.step(beta_true, gamma_true, u_true, v_init, eta_true, data)
        assert np.linalg.norm(vs - v_true)/ np.linalg.norm(v_true) < 5e-2

        # test fit
        eta_init = [np.random.rand()]
        alt_solver.fit(x_init=[beta_init, gamma_init, u_init, v_init, eta_init], data=data, options=dict(maxiter=5))

    





