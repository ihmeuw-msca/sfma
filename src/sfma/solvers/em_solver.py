import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

from anml.solvers.interface import CompositeSolver, Solver
from anml.solvers.base import IPOPTSolver, ClosedFormSolver

from sfma.data import Data
from sfma.models.base import LinearMarginal
from sfma.models.models import UModel, VModel


class EMSolver(CompositeSolver):

    def __init__(self, solvers_list: Optional[List[Solver]] = None):
        super().__init__(solvers_list)
        if len(solvers_list) != 2:
            self.beta_solver = IPOPTSolver(LinearMarginal())
            self.v_solver = ClosedFormSolver(VModel())
        else:
            self.beta_solver, self.v_solver = solvers_list
        self.solvers = [self.beta_solver, self.v_solver]

    def _set_params(self, data: Data):
        self.beta_solver.model.param_set = data.params[0]
        self.v_solver.model.param_set = data.params[1]

    def step(self, data, verbose=True):
        # fitting betas
        data.y = data.obs + self.v_solver.model.forward(self.vs_curr)
        self.beta_solver.fit(self.betas_curr, data, options=dict(solver_options=dict(maxiter=100)))
        self.betas_curr = self.beta_solver.x_opt
        
        # fitting vs
        self.v_solver.model.gammas = [self.eta_curr] 
        data.y = self.beta_solver.model.forward(self.betas_curr) - data.obs
        self.v_solver.fit(self.vs_curr, data)
        self.vs_curr = self.v_solver.x_opt

        # fitting eta
        sigma2_v = (data.obs_se**2 * self.eta_curr) / (self.eta_curr + data.obs_se**2)
        vs_cond = self.vs_curr + np.sqrt(sigma2_v * 2 / np.pi)
        sigma2_v_cond = (1 - 2 / np.pi) * sigma2_v
        self.eta_curr = [np.mean(vs_cond**2 + sigma2_v_cond)]

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
            self.betas_curr = x[0]
            self.vs_curr = x[1]
            self.eta_curr = x[2]
        else:
            self._x_curr = None

    def print_x_curr(self):
        print(f'betas = {self.betas_curr}')
        print(f'vs = {self.vs_curr}')
        print(f'eta = {self.eta_curr}')

    def fit(self, x_init: List[np.ndarray], data: Data, set_params: bool = True, verbose=True, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        self.x_curr = x_init 
        
        data.y = deepcopy(data.obs)
        if set_params:
            self._set_params(data)
        
        self.errors_hist = []
        itr = 0
        while itr < options['maxiter']:
            if verbose:
                print('-' * 10, f'iter {itr}', '-' * 10)
            self.step(data, verbose)
            self.errors_hist.append(self.error(data))
            if verbose:
                print(f'error = {self.errors_hist[-1]}')
            itr += 1

        self.x_opt = self.x_curr
        self.fun_val_opt = self.error(data)

    def predict(self):
        return self.beta_solver.predict() - self.v_solver.predict()

    def forward(self, x):
        betas, vs, _ = x
        return self.beta_solver.model.forward(betas) - self.v_solver.model.forward(vs)
    
    def error(self, data: Data, x = None):
        if x is None:
            x = self.x_curr
        return np.mean((data.obs - self.forward(x))**2 / data.obs_se**2)