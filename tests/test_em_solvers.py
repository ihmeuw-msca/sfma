from mock import Mock, patch, PropertyMock
import numpy as np
import pytest

from anml.parameter.prior import GaussianPrior, Prior
from anml.solvers.base import ScipyOpt, ClosedFormSolver

from sfma.models.maximal import UModel, BetaModel
from sfma.solvers.em_solver import EMSolver


@pytest.fixture
def sfa_inputs():
    np.random.seed(42)
    n_data, n_beta, n_v = 200, 3 , 5
    eta_true = 1.0
    beta_true = np.random.randn(n_beta)
    v_true = np.maximum(0, np.random.randn(n_v) * eta_true)
    X = np.random.randn(n_data, n_beta)
    Z_v = np.kron(np.identity(n_v), np.ones((n_data // n_v, 1)))
    sigma = 0.05 * np.ones(n_data)
    e = np.random.randn(n_data) * sigma
    y = np.dot(X, beta_true) - np.dot(Z_v, v_true) + e

    # mock data 
    mock_data = Mock()
    mock_data.obs = y 
    mock_data.y = y
    mock_data.obs_se = sigma
    mock_data.sigma2 = sigma**2

    return mock_data, X, Z_v, beta_true, eta_true, v_true


@patch.object(UModel, 'init_model', lambda: None)
@patch.object(BetaModel, 'init_model', lambda: None)
def test_alternating_solver(sfa_inputs):
    data, X, Z_v, beta_true, eta_true, v_true = sfa_inputs
    n_beta, n_v = len(beta_true), len(v_true)
    
    beta_model = BetaModel()
    beta_model.x_dim = n_beta
    beta_model.n_betas = n_beta
    beta_model.X = X
    beta_model.D = np.ones((n_v, 1))
    beta_model.prior_fun = lambda x: 0.0

    v_model = UModel()
    v_model.x_dim = n_v 
    v_model.Z = Z_v 
    v_model.D = np.ones((n_v, 1))
    v_model.gammas = [eta_true]

    em_solver = EMSolver(solvers_list=[ScipyOpt(beta_model), ClosedFormSolver(v_model)])

    beta_init = np.zeros(len(beta_true))
    v_init = np.random.rand(len(v_true))
    eta_init = [np.random.rand()]

    em_solver.x_curr = [beta_true, v_init, eta_true]
    em_solver.beta_solver.x_opt = beta_true
    em_solver.v_step(data)
    assert np.linalg.norm(em_solver.x_curr[1] - v_true)/ np.linalg.norm(v_true) < 5e-2

    # test fit
    em_solver.fit(x_init=[beta_init, v_init, eta_init], data=data, verbose=False, options=dict(maxiter=2))
