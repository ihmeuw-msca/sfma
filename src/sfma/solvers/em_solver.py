import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
import scipy

from anml.solvers.interface import Solver
from anml.solvers.base import IPOPTSolver, ClosedFormSolver, ScipyOpt

from sfma.data import Data
from sfma.models.marginal import LinearMarginal
from sfma.models.maximal import UModel, VModel
from sfma.solvers.base import IterativeSolver
from sfma.solvers.block_solver import BlockSolver


class EMSolver(BlockSolver):

    def __init__(self, solvers_list: List[Solver]):
        if len(solvers_list) != 2:
            raise RuntimeError(f'This EM solver only works for cases with beta solver and v solver.')
        super().__init__([solvers_list[0], None, solvers_list[1]])
        if not (self.v_solver.model.Z.shape[0] == self.v_solver.model.Z.shape[1] == np.sum(self.v_solver.model.Z)):
            raise RuntimeError('This solver only works for per datapoint inefficiency.')

    def v_step(self, data):
        data.y = self.beta_solver.predict() - data.obs
        eta_curr = self.x_curr[-1][0]
        
        self.v_solver.model.gammas = [eta_curr]
        self.v_solver.fit(self.x_curr[-2], data) # self.x_curr[-2] = v

        mu_v = eta_curr * data.y / (eta_curr + data.sigma2)
        sigma2_v = (data.sigma2 * eta_curr) / (eta_curr + data.sigma2)
       
        alpha = -mu_v / np.sqrt(sigma2_v)
        phi_alpha = 1/np.sqrt(2*np.pi) * np.exp(- alpha**2 / 2)
        Phi_alpha = (1 + scipy.special.erf(alpha / np.sqrt(2))) / 2
        z = 1 - Phi_alpha
        vs_curr = mu_v + phi_alpha * np.sqrt(sigma2_v) / z
        
        self.x_curr[-2] = vs_curr
        sigma2_v_expect = (1 + alpha * phi_alpha / z - (phi_alpha / z)**2) * sigma2_v
        eta_curr = np.mean(vs_curr**2 + sigma2_v_expect)
        self.x_curr[-1] = [eta_curr] # self.x_curr[-1] = eta
        self.v_solver.x_opt = vs_curr

        data.sigma2 = (np.sum(data.y**2) + np.sum(vs_curr**2 + sigma2_v_expect) -2 * np.dot(data.y, vs_curr)) / len(data.y) * np.ones(len(data.y))
        assert np.all(data.sigma2 > 0.0)

    def error(self, data):
        r = data.obs - self.beta_solver.predict()
        eta = self.x_curr[-1][0]
        z = np.sqrt(eta) * r / np.sqrt(2 * (eta + data.sigma2) * data.sigma2)
        Phi = 1 - scipy.special.erf(z)
        # import pdb; pdb.set_trace()
        return 0.5 * np.sum(r**2 / (eta + data.sigma2)) +  0.5 * np.sum(np.log(eta + data.sigma2)) - np.sum(np.log(Phi))
