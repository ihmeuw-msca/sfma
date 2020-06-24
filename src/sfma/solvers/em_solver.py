import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

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
        vs_mle = self.v_solver.x_opt

        # fitting eta
        sigma2_v = (data.sigma2 * self.eta_curr) / (self.eta_curr + data.sigma2)
        self.vs_curr = vs_mle + np.sqrt(sigma2_v * 2 / np.pi)
        sigma2_v_cond = (1 - 2 / np.pi) * sigma2_v
        self.eta_curr = [np.mean(self.vs_curr**2 + sigma2_v_cond)]

        data.sigma2 = np.ones(len(data.y)) * np.mean(np.dot(data.y, data.y + self.v_solver.model.forward(self.vs_curr)))

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
