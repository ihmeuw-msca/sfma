import numpy as np

from anml.parameter.parameter import ParameterSet
from anml.parameter.prior import GaussianPrior
from anml.parameter.utils import collect_priors

from sfma.models.base import LinearModel
from sfma.data import Data
from sfma.models.utils import build_linear_constraint


class LinearMaximal(LinearModel):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__(param_set_processed)
    
    def objective(self, x, data: Data):
        self._prerun_check(x)
        sigma2 = data.sigma2
        return np.sum((data.y - np.dot(self.design_matrix, x))**2 / (2*sigma2)) + self.prior_fun(x)


class BetaModel(LinearMaximal):

    def init_model(self):        
        self.x_dim = self._param_set.num_fe
        self.X = self._param_set.design_matrix_fe
        self.lb = self._param_set.lb_fe
        self.ub = self._param_set.ub_fe

        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self._param_set.constr_matrix_fe, self._param_set.constr_lb_fe, self._param_set.constr_ub_fe)
        ])
        self.prior_fun = collect_priors(self._param_set.fe_priors)

    @property
    def design_matrix(self):
        return self.X

    def closed_form_soln(self, data: Data):
        # only valid when no constraints, no bounds and no priors
        # used for sanity check
        x = np.linalg.solve(np.dot(self.X.T / data.sigma2, self.X), np.dot(self.X.T, data.y / data.sigma2))
        return x


class UModel(LinearMaximal):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__(param_set_processed)
        if param_set_processed is not None:
            if not all([isinstance(prior, GaussianPrior) for prior in param_set_processed.re_priors]):
                raise TypeError('Only Gaussian type priors allowed.')

    def init_model(self):
        self.x_dim = self._param_set.num_re
        self.Z = self._param_set.design_matrix_re
        self.D = self._param_set.re_var_padding
        self.lb = self._param_set.lb_re
        self.ub = self._param_set.ub_re

        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self._param_set.constr_matrix_re, self._param_set.constr_lb_re, self._param_set.constr_ub_re)
        ])
        self.gammas_padded = np.array([prior.std[0] for prior in self._param_set.re_priors])
        self._gammas = None

    @property
    def gammas(self):
        return self._gammas

    @gammas.setter
    def gammas(self, new_gammas):
        self._gammas = new_gammas
        self.gammas_padded = np.squeeze(np.dot(self.D, new_gammas))

    @property
    def design_matrix(self):
        return self.Z

    @property
    def prior_fun(self):
        self._prior_fun = lambda x: np.sum(x**2/self.gammas_padded) / 2
        return self._prior_fun 

    def closed_form_soln(self, data: Data):
        sigma2 = data.sigma2
        return np.linalg.solve(
            np.dot(self.Z.T / sigma2, self.Z) + np.diag(1.0/self.gammas_padded), 
            np.dot(self.Z.T, data.y / sigma2),
        )
    

class VModel(UModel):

    def closed_form_soln(self, data: Data):
        soln = super().closed_form_soln(data)
        return np.maximum(0, soln)
        