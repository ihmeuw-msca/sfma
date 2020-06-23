import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

from anml.solvers.interface import CompositeSolver, Solver
from anml.solvers.base import ScipyOpt, ClosedFormSolver

from sfma.data import Data
from sfma.models import LinearMixedEffectsMarginal, UModel, VModel


class AlternatingSolver(CompositeSolver):

    def __init__(self, solvers_list: Optional[List[Solver]] = None):
        super().__init__(solvers_list)
        if len(solvers_list) < 3:
            self.lme_solver = ScipyOpt(LinearMixedEffectsMarginal())
            self.u_solver = ClosedFormSolver(UModel())
            self.v_solver = ClosedFormSolver(VModel())
        else:
            self.lme_solver, self.u_solver, self.v_solver = solvers_list[:3]
        self.solvers = [self.lme_solver, self.u_solver, self.v_solver]

    def _set_params(self, data: Data):
        lme_param_set = data.params[0]
        u_param_set = data.params[1]
        v_param_set = data.params[2]

        self.lme_solver.model.param_set = lme_param_set
        self.u_solver.model.param_set = u_param_set
        self.v_solver.model.param_set = v_param_set

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
        self.eta_curr = [np.std(self.vs_curr) ** 2]

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
            self.betas_curr = x[0]
            self.gammas_curr = x[1]
            self.us_curr = x[2]
            self.vs_curr = x[3]
            self.eta_curr = x[4]
        else:
            self._x_curr = None

    def print_x_curr(self):
        print(f'betas = {self.betas_curr}')
        print(f'gammas = {self.gammas_curr}')
        print(f'us = {self.us_curr}')
        print(f'vs = {self.vs_curr}')
        print(f'eta = {self.eta_curr}')

    def fit(self, x_init: List[np.ndarray], data: Data, set_params: bool = True, verbose=True, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        self.betas_curr, self.gammas_curr, self.us_curr, self.vs_curr, self.eta_curr = x_init 
        
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

        self.x_opt = [self.betas_curr, self.gammas_curr, self.us_curr, self.vs_curr, self.eta_curr]
        self.fun_val_opt = self.error(data)

    def predict(self):
        return self.lme_solver.predict() + self.u_solver.predict() - self.v_solver.predict()

    def forward(self, x):
        betas, gammas, us, vs, _ = x
        return self.lme_solver.model.forward(np.hstack((betas, gammas))) + self.u_solver.model.forward(us) - self.v_solver.model.forward(vs)
    
    def error(self, data: Data, x = None):
        if x is None:
            x = self.x_curr
        return np.mean((data.obs - self.forward(x))**2 / data.obs_se**2)







        
