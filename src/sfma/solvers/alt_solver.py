import numpy as np
from typing import List, Dict, Optional

from anml.solvers.base import IPOPTSolver, ClosedFormSolver
from anml.solvers.interface import Solver

from sfma.data import Data
from sfma.models.marginal import LinearMarginal
from sfma.models.maximal import UModel, VModel
from sfma.solvers.base import IterativeSolver


class AlternatingSolver(IterativeSolver):

    def __init__(self, solvers_list: Optional[List[Solver]] = None):
        super().__init__(solvers_list)
        if len(solvers_list) != 3:
            self.lme_solver = IPOPTSolver(LinearMarginal())
            self.u_solver = ClosedFormSolver(UModel())
            self.v_solver = ClosedFormSolver(VModel())
            self.solvers = [self.lme_solver, self.u_solver, self.v_solver]
        else:
            self.lme_solver, self.u_solver, self.v_solver = solvers_list

    def step(self, data, verbose=True):
        # fitting betas
        data.y = data.obs + self.v_solver.model.forward(self.vs_curr)
        beta_gamma = np.hstack((self.betas_curr, self.gammas_curr))
        self.lme_solver.fit(beta_gamma, data, options=dict(solver_options=dict(maxiter=100)))
        beta_gamma = self.lme_solver.x_opt
        self.betas_curr = beta_gamma[:len(self.betas_curr)]
        self.gammas_curr = beta_gamma[len(self.betas_curr):]
        
        # fitting us
        self.u_solver.model.gammas = self.gammas_curr
        data.y -= self.lme_solver.model.forward(beta_gamma)
        self.u_solver.fit(self.us_curr, data)
        self.us_curr = self.u_solver.x_opt
        
        # fitting vs
        self.v_solver.model.gammas = [self.eta_curr] 
        data.y = self.lme_solver.model.forward(beta_gamma) + self.u_solver.model.forward(self.us_curr) - data.obs
        self.v_solver.fit(self.vs_curr, data)
        self.vs_curr = self.v_solver.x_opt

        # fitting eta
        self.eta_curr = [np.dot(self.v_solver.model.D.T, self.vs_curr**2) / np.sum(self.v_solver.model.D, axis=0)]

        if verbose:
            self.print_x_curr()

    @property 
    def x_curr(self):
        self._x_curr = [self.betas_curr, self.gammas_curr, self.us_curr, self.vs_curr, self.eta_curr]
        return self._x_curr

    @x_curr.setter
    def x_curr(self, x):
        if x is not None:
            assert len(x) == 5
            self._x_curr = x
            self.betas_curr, self.gammas_curr, self.us_curr, self.vs_curr, self.eta_curr = x
        else:
            self._x_curr = None

    def print_x_curr(self):
        print(f'betas = {self.betas_curr}')
        print(f'gammas = {self.gammas_curr}')
        print(f'us = {self.us_curr}')
        print(f'vs = {self.vs_curr}')
        print(f'eta = {self.eta_curr}')

    def predict(self):
        return self.lme_solver.predict() + self.u_solver.predict() - self.v_solver.predict()

    def forward(self, x):
        betas, gammas, us, vs, _ = x
        return self.lme_solver.model.forward(np.hstack((betas, gammas))) + self.u_solver.model.forward(us) - self.v_solver.model.forward(vs)








        
