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
        if len(self.solvers) < 3:
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

    def step(self, betas, gammas, us, vs, eta, data):
        data.y = data.obs + np.dot(self.v_solver.model.Z, vs)
        beta_gamma = np.hstack((betas, gammas))
        self.lme_solver.fit(beta_gamma, data, options=dict(solver_options=dict(maxiter=100)))
        beta_gamma = self.lme_solver.x_opt
        betas = beta_gamma[:len(betas)]
        gammas = beta_gamma[len(betas):]
        # fitting us
        self.u_solver.gammas = gammas
        data.y -= np.dot(self.lme_solver.model.X, betas)
        self.u_solver.fit(us, data)
        us = self.u_solver.x_opt
        # fitting vs
        self.v_solver.gammas = [eta] 
        data.y = np.dot(self.lme_solver.model.X, betas) + np.dot(self.u_solver.model.Z, us) - data.obs 
        self.v_solver.fit(vs, data)
        vs = self.v_solver.x_opt
        # fitting eta
        eta = [np.std(vs) ** 2]
        return betas, gammas, us, vs, eta

    def fit(self, x_init: List[np.ndarray], data: Data, set_params: bool = True, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        betas, gammas, us, vs, eta = x_init 
        data.y = deepcopy(data.obs)
        if set_params:
            self._set_params(data)
        
        self.errors_hist = []
        itr = 0
        while itr < options['maxiter']:
            betas, gammas, us, vs, eta = self.step(betas, gammas, us, vs, eta, data)
            self.errors_hist.append(self.error([betas, gammas, us, vs, eta], data))
            if 'verbose' in options and options['verbose']:
                print('-------------')
                print(f'iter {itr} \t error = {self.errors_hist[-1]}')
                print(f'betas = {betas}')
                print(f'gammas = {gammas}')
                print(f'us = {us}')
                print(f'vs = {vs}')
                print(f'eta = {eta}')
            itr += 1
        
        self.beta_final = betas
        self.gamma_final = gammas
        self.u_final = us 
        self.v_final = vs 
        self.eta_final = eta 

        self.x_opt = [self.beta_final, self.gamma_final, self.u_final, self.v_final, self.eta_final]
        self.fun_val_opt = self.error(self.x_opt, data)

    def predict(self, x):
        betas, _, us, vs, _ = x
        return self.lme_solver.model.predict(betas) + self.u_solver.model.predict(us) - self.v_solver.model.predict(vs)

    def error(self, x, data: Data):
        return np.mean((data.obs - self.predict(x))**2 / data.obs_se**2)







        
