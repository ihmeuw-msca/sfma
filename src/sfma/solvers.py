import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

from anml.solvers.interface import CompositeSolver, Solver
from anml.solvers.base import ScipyOpt, ClosedFormSolver

from sfma.data import Data
from sfma.models import LinearMixedEffectsMarginal, RandomEffectsOnly


class AlternatingSolver(CompositeSolver):

    def __init__(self, solvers_list: Optional[List[Solver]] = None):
        super().__init__(solvers_list)
        if len(self.solvers) < 3:
            self.lme_solver = ScipyOpt(LinearMixedEffectsMarginal())
            self.u_solver = ScipyOpt(RandomEffectsOnly())
            self.v_solver = ScipyOpt(RandomEffectsOnly())

    def fit(self, x_init: List[np.ndarray], data: Data, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        betas, gammas, us, vs, eta = x_init 
        beta_gamma = np.hstack((betas, gammas))
        obs = deepcopy(data.obs)

        lme_param_set = data.params[0]
        n_betas = lme_param_set.num_fe

        u_param_set = data.params[1]
        v_param_set = data.params[2]

        self.lme_solver.model.param_set = lme_param_set
        
        itr, err = 0, 1.0
        while itr < options['maxiter'] or err > options['tol']:
            # fitting betas and gammas
            data.y = obs + np.dot(self.v_solver.model.Z, vs)
            self.lme_solver.fit(beta_gamma, data)
            beta_gamma = self.lme_solver.x_opt
            betas = beta_gamma[:n_betas]
            gammas = beta_gamma[n_betas:]

            # fitting us
            u_variable.re_prior.std = np.dot(self.u_solver.D, gammas)
            data.y -= np.dot(self.lme_solver.model.X, betas)
            self.u_solver.model.variable = u_variable
            self.u_solver.fit(us, data)
            us = self.u_solver.x_opt

            # fitting vs
            v_variable.re_prior.std = eta 
            data.y = np.dot(self.lme_solver.model.X, betas) + np.dot(self.u_solver.model.Z, us) - obs 
            self.v_solver.model.variable = v_variable
            self.v_solver.fit(vs, data)
            vs = self.v_solver.x_opt

            # fitting eta
            eta = [np.std(vs)]






        
