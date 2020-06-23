import numpy as np
import pandas as pd

from anml.parameter.parameter import ParameterSet
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
        self.x_dim = self.n_betas + self.n_gammas
        
        self.X = self._param_set.design_matrix_fe
        self.Z = self._param_set.design_matrix_re
        self.D = self._param_set.re_var_padding
        
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
        gammas = x[self.n_betas:]

        Sigma = np.diag(data.obs_se)
        Gamma = np.diag(np.dot(self.D, gammas))

        V = Sigma**2 + np.dot(self.Z,np.dot(Gamma, self.Z.T))
        r = data.y - self.X.dot(betas)
        return 0.5 * np.dot(r, np.linalg.solve(V, r)) + np.prod(np.linalg.slogdet(V)) * 0.5 + self.prior_fun(x)

    @property
    def design_matrix(self):
        return self.X

    def forward(self, x, mat=None):
        betas = x[:self.n_betas]
        return super().forward(betas, mat)