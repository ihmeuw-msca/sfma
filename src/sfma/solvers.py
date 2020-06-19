import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

from anml.solvers.interface import CompositeSolver, Solver
from anml.solvers.base import ScipyOpt, ClosedFormSolver

from sfma.data import Data
from sfma.models import LinearMixedEffectsMarginal, GaussianRandomEffects


class AlternatingSolver(CompositeSolver):

    def __init__(self, solvers_list: Optional[List[Solver]] = None):
        super().__init__(solvers_list)
        if len(self.solvers) < 3:
            self.lme_solver = ScipyOpt(LinearMixedEffectsMarginal())
            self.u_solver = ClosedFormSolver(GaussianRandomEffects())
            self.v_solver = ClosedFormSolver(GaussianRandomEffects())

    def fit(self, x_init: List[np.ndarray], data: Data, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        betas, gammas, us, vs, eta = x_init 
        beta_gamma = np.hstack((betas, gammas))
        obs = deepcopy(data.obs)

        lme_param_set = data.params[0]
        n_betas = lme_param_set.num_fe
        u_param_set = data.params[1]
        v_param_set = data.params[2]

        self.lme_solver.model.param_set = lme_param_set
        self.u_solver.model.param_set = u_param_set
        self.v_solver.model.param_set = v_param_set
        
        itr = 0
        while itr < options['maxiter']:
            # fitting betas and gammas
            data.y = obs + np.dot(self.v_solver.model.Z, vs)
            self.lme_solver.fit(beta_gamma, data)
            beta_gamma = self.lme_solver.x_opt
            betas = beta_gamma[:n_betas]
            gammas = beta_gamma[n_betas:]

            # fitting us
            self.u_solver.gammas = gammas
            data.y -= np.dot(self.lme_solver.model.X, betas)
            self.u_solver.fit(us, data)
            us = self.u_solver.x_opt

            # fitting vs
            self.v_solver.gammas = [eta] 
            data.y = np.dot(self.lme_solver.model.X, betas) + np.dot(self.u_solver.model.Z, us) - obs 
            self.v_solver.fit(vs, data)
            vs = self.v_solver.x_opt

            # fitting eta
            eta = [np.std(vs)]

            itr += 1






        
