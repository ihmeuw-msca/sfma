import numpy as np
import pandas as pd

from anml.parameter.parameter import ParameterSet
from anml.parameter.prior import Prior
from anml.parameter.utils import collect_priors

from sfma.data import Data
from sfma.models.base import LinearModel
from sfma.models.utils import build_linear_constraint


class LinearMarginal(LinearModel):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__(param_set_processed)

    def init_model(self):
        self.n_betas = self._param_set.num_fe
        self.n_gammas = self._param_set.num_re_var
        
        self.X = self._param_set.design_matrix_fe
        self.Z = self._param_set.design_matrix_re
        self.D = self._param_set.re_var_padding

    def _loss(self, betas, gammas, sigma2, y):
        Sigma2 = np.diag(sigma2)
        Gamma = np.diag(np.dot(self.D, gammas))

        V = Sigma2 + np.dot(self.Z,np.dot(Gamma, self.Z.T))
        r = y - self.X.dot(betas)
        return 0.5 * np.dot(r, np.linalg.solve(V, r)) + np.prod(np.linalg.slogdet(V)) * 0.5

    @property
    def design_matrix(self):
        return self.X

    def forward(self, x, mat=None):
        betas = x[:self.n_betas]
        return super().forward(betas, mat)


class BetaGammaModel(LinearMarginal):

    def init_model(self):
        super().init_model()
        self.x_dim = self.n_betas + self.n_gammas
        self.lb = np.hstack((self._param_set.lb_fe, self._param_set.lb_re_var))
        self.ub = np.hstack((self._param_set.ub_fe, self._param_set.ub_re_var)) 
        
        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self._param_set.constr_matrix_fe, self._param_set.constr_lb_fe, self._param_set.constr_ub_fe),
            (self._param_set.constr_matrix_re_var, self._param_set.constr_lb_re_var, self._param_set.constr_ub_re_var),
        ])
        self.prior_fun = collect_priors(self._param_set.fe_priors + self._param_set.re_var_priors) 
    
    def objective(self, x, data: Data):
        self._prerun_check(x)
        betas = x[:self.n_betas]
        gammas = x[self.n_betas:self.n_betas + self.n_gammas]
        
        return self._loss(betas, gammas, data.sigma2, data.y) + self.prior_fun(x)


class BetaGammaSigmaModel(LinearMarginal):

    def __init__(self, param_set_processed: ParameterSet = None, sigma2_prior: Prior = None):
        if sigma2_prior is not None:
            self.sigma2_prior = sigma2_prior
        else:
            self.sigma2_prior = Prior(lower_bound=[0.0], upper_bound=[np.inf])
        super().__init__(param_set_processed)

    def init_model(self):
        super().init_model()
        self.x_dim = self.n_betas + self.n_gammas + 1
        self.lb = np.hstack((self._param_set.lb_fe, self._param_set.lb_re_var, self.sigma2_prior.lower_bound))
        self.ub = np.hstack((self._param_set.ub_fe, self._param_set.ub_re_var, self.sigma2_prior.upper_bound)) 
        
        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self._param_set.constr_matrix_fe, self._param_set.constr_lb_fe, self._param_set.constr_ub_fe),
            (self._param_set.constr_matrix_re_var, self._param_set.constr_lb_re_var, self._param_set.constr_ub_re_var),
        ])
        if self.C is not None:
            self.C = np.hstack((self.C, np.zeros((len(self.C), 1))))
        self.prior_fun = collect_priors(self._param_set.fe_priors + self._param_set.re_var_priors + [self.sigma2_prior]) 

    def objective(self, x, data: Data):
        self._prerun_check(x)
        betas = x[:self.n_betas]
        gammas = x[self.n_betas:self.n_betas + self.n_gammas]
        sigma2 = np.ones(len(data.y)) * x[-1]

        return self._loss(betas, gammas, sigma2, data.y) + self.prior_fun(x)

