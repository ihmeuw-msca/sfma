import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
import scipy

from anml.solvers.interface import Solver
from anml.solvers.base import IPOPTSolver, ClosedFormSolver

from sfma.data import Data
from sfma.models.marginal import LinearMarginal
from sfma.models.maximal import UModel, VModel
from sfma.solvers.base import IterativeSolver


class EMSolver(IterativeSolver):

    def __init__(self, solvers_list: Optional[List[Solver]] = None):
        super().__init__(solvers_list)
        if len(solvers_list) != 2:
            self.beta_solver = IPOPTSolver(LinearMarginal())
            self.v_solver = ClosedFormSolver(VModel())
            self.solvers = [self.beta_solver, self.v_solver]
        else:
            self.beta_solver, self.v_solver = solvers_list

    def step(self, data, verbose=True):
        # fitting betas
        data.y = data.obs + self.v_solver.model.forward(self.vs_curr)
        self.beta_solver.fit(self.betas_curr, data, options=dict(solver_options=dict(maxiter=100)))
        self.betas_curr = self.beta_solver.x_opt
        
        # fitting vs
        self.v_solver.model.gammas = [self.eta_curr] 
        data.y = self.beta_solver.model.forward(self.betas_curr) - data.obs
        self.v_solver.fit(self.vs_curr, data)
        mu_v = self.v_solver.x_opt

        # mu_v = self.eta_curr * data.y / (self.eta_curr + data.sigma2)
        # sigma2_v = (data.sigma2 * self.eta_curr) / (self.eta_curr + data.sigma2)
        
        # fitting eta
        # note: computation for expectations follow from 
        # https://en.wikipedia.org/wiki/Truncated_normal_distribution
        sigma2_v = 1 / (np.diag(np.dot(self.v_solver.model.Z.T / data.sigma2, self.v_solver.model.Z)) + 1.0/self.v_solver.model.gammas_padded)
        alpha = -mu_v / np.sqrt(sigma2_v)
        phi_alpha = 1/np.sqrt(2*np.pi) * np.exp(- alpha**2 / 2)
        Phi_alpha = (1 + scipy.special.erf(alpha / np.sqrt(2))) / 2
        z = 1 - Phi_alpha
        self.vs_curr = mu_v + phi_alpha * np.sqrt(sigma2_v) / z
        sigma2_v_expect = (1 + alpha * phi_alpha / z - (phi_alpha / z)**2) * sigma2_v
        self.eta_curr = [np.mean(self.vs_curr**2 + sigma2_v_expect)]

        data.sigma2 = (
            np.sum(data.y**2) + 
            np.sum(np.diag(np.dot(self.v_solver.model.Z.T, self.v_solver.model.Z)) * (self.vs_curr**2 + sigma2_v_expect)) -
            2 * np.dot(data.y, np.dot(self.v_solver.model.Z, self.vs_curr))
        ) / len(data.y)

        if verbose:
            self.print_x_curr()

    @property 
    def x_curr(self):
        self._x_curr = [self.betas_curr, self.vs_curr, self.eta_curr]
        return self._x_curr

    @x_curr.setter
    def x_curr(self, x):
        if x is not None:
            assert len(x) == 3
            self._x_curr = x
            self.betas_curr, self.vs_curr, self.eta_curr = x
        else:
            self._x_curr = None

    def print_x_curr(self):
        print(f'betas = {self.betas_curr}')
        print(f'vs = {self.vs_curr}')
        print(f'eta = {self.eta_curr}')

    def predict(self):
        return self.beta_solver.predict() - self.v_solver.predict()

    def forward(self, x):
        betas, vs, _ = x
        return self.beta_solver.model.forward(betas) - self.v_solver.model.forward(vs)
