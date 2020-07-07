from mock import Mock, patch, PropertyMock
import numpy as np
import pytest

from anml.parameter.prior import GaussianPrior, Prior
from anml.solvers.base import ScipyOpt, ClosedFormSolver

from sfma.models.marginal import BetaGammaModel
from sfma.models.maximal import UModel, VModel
from sfma.solvers.block_solver import BlockSolver


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
    mock_data.y = y
    mock_data.obs_se = sigma
    mock_data.sigma2 = sigma**2

    return mock_data, X, Z_u, Z_v, beta_true, gamma_true, eta_true, u_true, v_true


@patch.object(UModel, 'init_model', lambda: None)
@patch.object(BetaGammaModel, 'init_model', lambda: None)
def test_alternating_solver(sfa_inputs):
    data, X, Z_u, Z_v, beta_true, gamma_true, eta_true, u_true, v_true = sfa_inputs
    n_beta, n_gamma, n_u, n_v = len(beta_true), 1, len(u_true), len(v_true)
    
    lme_model = BetaGammaModel()
    lme_model.x_dim = n_beta + n_gamma
    lme_model.n_betas = n_beta
    lme_model.n_gammas = n_gamma
    lme_model.X = X
    lme_model.Z = Z_u
    lme_model.D = np.ones((n_u, 1))
    lme_model.prior_fun = lambda x: 0.0

    u_model = UModel()
    u_model.x_dim = n_u
    u_model.Z = Z_u 
    u_model.D = np.ones((n_u, 1))
    u_model.gammas = [gamma_true]

    v_model = VModel()
    v_model.x_dim = n_v 
    v_model.Z = Z_v 
    v_model.D = np.ones((n_v, 1))
    v_model.gammas = [eta_true]

    blk_solver = BlockSolver(solvers_list=[ScipyOpt(lme_model), ClosedFormSolver(u_model), ClosedFormSolver(v_model)])

    beta_init = np.zeros(len(beta_true))
    gamma_init = [np.random.rand()]
    u_init = np.zeros(len(u_true))
    v_init = np.zeros(len(v_true)) + 0.1
    eta_init = [np.random.rand()]

    blk_solver.x_curr = [beta_init, gamma_init, u_true, v_true, eta_true]
    blk_solver.v_solver.x_opt = v_true
    blk_solver.beta_step(data)
    assert np.linalg.norm(blk_solver.x_curr[0] - beta_true)/ np.linalg.norm(beta_true) < 2e-2

    blk_solver.x_curr = [beta_true, gamma_true, u_init, v_true, eta_true]
    blk_solver.u_step(data)
    assert np.linalg.norm(blk_solver.x_curr[2] - u_true)/ np.linalg.norm(u_true) < 2e-2

    blk_solver.x_curr = [beta_true, gamma_true, u_true, v_init, eta_true]
    blk_solver.v_step(data)
    assert np.linalg.norm(blk_solver.x_curr[-2] - v_true)/ np.linalg.norm(v_true) < 5e-2

    # test fit
    blk_solver.fit(x_init=[beta_init, gamma_init, u_init, v_init, eta_init], data=data, verbose=False, options=dict(maxiter=2))