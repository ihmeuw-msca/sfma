import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

from anml.solvers.base import IPOPTSolver, ClosedFormSolver, ScipyOpt
from anml.solvers.interface import Solver

from sfma.data import Data
from sfma.models.marginal import BetaGammaModel, BetaGammaSigmaModel
from sfma.models.maximal import UModel, VModel, BetaModel
from sfma.solvers.base import IterativeSolver


class BlockSolver(IterativeSolver):

    def __init__(self, solvers_list: List[Solver]):
        if len(solvers_list) != 3:
            raise ValueError(f'Must have three solvers for beta, u, and v blocks respectively. Can pass in None for u solver if u does not exist.')
        self.variables = []
        self.blocks_divider = []
        self.beta_solver, self.u_solver, self.v_solver = solvers_list
        if self.beta_solver is None or self.v_solver is None:
            raise ValueError(f'Must have valid solvers for beta and v in SFA.')
        if self.beta_solver.model.__class__.__name__ == 'BetaModel':
            self.variables.append('beta')
            self.n_betas = self.beta_solver.model.x_dim
            self.blocks_divider.append(1)
        elif self.beta_solver.model.__class__.__name__ == 'BetaGammaModel':
            self.variables.extend(['beta', 'gamma'])
            self.n_betas = self.beta_solver.model.n_betas
            self.n_gammas = self.beta_solver.model.n_gammas
            self.blocks_divider.append(2)
        elif self.beta_solver.model.__class__.__name__ == 'BetaGammaSigmaModel':
            self.variables.extend(['beta', 'gamma', 'sigma2'])
            self.n_betas = self.beta_solver.model.n_betas
            self.n_gammas = self.beta_solver.model.n_gammas
            self.blocks_divider.append(3)        
        if self.u_solver is not None:
            self.variables.append('u')
            self.blocks_divider.append(self.blocks_divider[-1] + 1)
        else:
            self.blocks_divider.append(self.blocks_divider[-1])
        self.variables.extend(['v', 'eta'])
        self.n_blocks = len(self.variables)
        super().__init__([solver for solver in solvers_list if solver is not None])

    def beta_step(self, data):
        data.y = data.obs + self.v_solver.predict()
        self.beta_solver.fit(np.hstack(self.x_curr[:self.blocks_divider[0]]), data, options=dict(solver_options=dict(maxiter=100)))
        self.x_curr[0] = self.beta_solver.x_opt[:self.n_betas]
        if 'gamma' in self.variables:
            self.x_curr[1] = self.beta_solver.x_opt[self.n_betas: self.n_betas + self.n_gammas]
            self.u_solver.model.gammas = self.x_curr[1]
        if 'sigma2' in self.variables:
            self.x_curr[2] = [self.beta_solver.x_opt[-1]]
            data.sigma2 = np.ones(len(data.y)) * self.x_curr[2][0]

    def u_step(self, data):
        data.y -= self.beta_solver.predict()
        self.u_solver.fit(self.x_curr[self.blocks_divider[0]], data)
        self.x_curr[self.blocks_divider[0]] = self.u_solver.x_opt

    def v_step(self, data):
        if self.u_solver is not None:
            data.y = self.beta_solver.predict() + self.u_solver.predict() - data.obs
        else:
            data.y = self.beta_solver.predict() - data.obs
        self.v_solver.model.gammas = self.x_curr[-1]
        self.v_solver.fit(self.x_curr[self.blocks_divider[1]], data)
        vs_curr = self.v_solver.x_opt
        self.x_curr[self.blocks_divider[1]] = vs_curr
        self.x_curr[-1] = np.dot(self.v_solver.model.D.T, vs_curr**2) / np.sum(self.v_solver.model.D, axis=0)

    def step(self, data, verbose=False):
        self.beta_step(data)
        if self.u_solver is not None:
            self.u_step(data)
        self.v_step(data)
        if verbose:
            self.print_variables()

    def print_variables(self):
        for name, value in zip(self.variables, self.x_curr):
            print(name, '=', value)

    def fit(self, x_init: List[np.ndarray], data: Data, verbose=True, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        if len(x_init) != self.n_blocks:
            raise ValueError(f'Lengh of x_init = {len(x_init)} is not equal to the number of variable blocks = {self.n_blocks}.')
        self.x_curr = x_init
        self.v_solver.x_opt = self.x_curr[self.blocks_divider[1]]
        super().fit(x_init, data, verbose, options)

    def predict(self):
        if self.u_solver is None:
            return self.beta_solver.predict() - self.v_solver.predict()
        else:
            return self.beta_solver.predict() + self.u_solver.predict() - self.v_solver.predict()

    def forward(self, x):
        if self.u_solver is None:
            return self.beta_solver.model.forward(np.hstack(x[:self.blocks_divider[0]])) - self.v_solver.model.forward(x[self.blocks_divider[1]])
        else:
            return (
                self.beta_solver.model.forward(np.hstack(x[:self.blocks_divider[0]])) + 
                self.u_solver.model.forward(x[self.blocks_divider[0]])- 
                self.v_solver.model.forward(x[self.blocks_divider[1]])
            )








        
