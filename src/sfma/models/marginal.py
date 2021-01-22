import numpy as np
import pandas as pd
import scipy

from anml.parameter.parameter import ParameterSet
from anml.parameter.prior import Prior
from anml.parameter.utils import collect_priors

from sfma.data import Data
from sfma.models.base import LinearModel
from sfma.models.utils import build_linear_constraint
from sfa_utils.npufunc import log_erfc


class MarginalModel(LinearModel):
        
    def __init__(self, param_set_processed: ParameterSet = None, eta_prior: Prior = None):
        if eta_prior is not None:
            self.eta_prior = eta_prior
        else:
            self.eta_prior = Prior(lower_bound=[0.0], upper_bound=[np.inf])
        super().__init__(param_set_processed)

    @property 
    def design_matrix(self):
        return self.X

    def _loss(self, betas, gamma, eta, data):
        r = data.y - self.X.dot(betas)
        eta = np.sqrt(eta**2)
        V = gamma + eta + data.sigma2
        z = np.sqrt(eta) * r / np.sqrt(2 * V * (gamma + data.sigma2))
        logPhi = log_erfc(z)
        return np.mean(r**2 / (2 * V) + 0.5 * np.log(V) - logPhi)

    def forward(self, x, X=None):
        return super().forward(x[:self.n_betas], X)


class SimpleBetaGammaEtaModel(MarginalModel):

    def init_model(self):
        self.n_betas = self._param_set.num_fe
        self.n_gammas = self._param_set.num_re_var
        assert self.n_gammas == 1
        
        self.X = self._param_set.design_matrix_fe
        self.Z = self._param_set.design_matrix_re
        assert all([np.sum(row) == 1 for row in self.Z]) # identity matrix

        self.x_dim = self.n_betas + self.n_gammas + 1
        self.lb = np.hstack((self._param_set.lb_fe, self._param_set.lb_re_var, self.eta_prior.lower_bound))
        self.ub = np.hstack((self._param_set.ub_fe, self._param_set.ub_re_var, self.eta_prior.upper_bound)) 
        
        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self._param_set.constr_matrix_fe, self._param_set.constr_lb_fe, self._param_set.constr_ub_fe),
            (self._param_set.constr_matrix_re_var, self._param_set.constr_lb_re_var, self._param_set.constr_ub_re_var),
        ])
        
        if self.C is not None:
            self.C = np.hstack((self.C, np.zeros((len(self.C), 1))))
        
        self.prior_fun = collect_priors(self._param_set.fe_priors + self._param_set.re_var_priors + [self.eta_prior])

    def objective(self, x, data: Data):
        self._prerun_check(x)
        betas = x[:self.n_betas]
        gamma = x[-2]
        eta = x[-1]
        return self._loss(betas, gamma, eta, data)

        
class SimpleBetaEtaModel(MarginalModel):

    def init_model(self):
        self.n_betas = self._param_set.num_fe
        
        self.X = self._param_set.design_matrix_fe

        self.x_dim = self.n_betas + 1
        self.lb = np.hstack((self._param_set.lb_fe, self.eta_prior.lower_bound))
        self.ub = np.hstack((self._param_set.ub_fe, self.eta_prior.upper_bound)) 
        
        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self._param_set.constr_matrix_fe, self._param_set.constr_lb_fe, self._param_set.constr_ub_fe),
        ])
        
        if self.C is not None:
            self.C = np.hstack((self.C, np.zeros((len(self.C), 1))))
        
        self.prior_fun = collect_priors(self._param_set.fe_priors + [self.eta_prior])

    def objective(self, x, data: Data):
        self._prerun_check(x)
        betas = x[:self.n_betas]
        eta = x[-1]
        return self._loss(betas, 0, eta, data)


