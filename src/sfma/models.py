# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import Bounds, LinearConstraint
from typing import List, Tuple

from anml.models.interface import Model
from anml.parameter.parameter import ParameterSet, Parameter
from anml.parameter.prior import GaussianPrior, Prior
from anml.parameter.variables import Variable, Intercept
from anml.parameter.utils import combine_constraints, collect_priors

from sfma.data import Data


class Base(Model):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__()
        if param_set_processed is not None:
            self.param_set = param_set_processed
        else:
            self._param_set = None

    def _prerun_check(self, x):
        if self._param_set is None:
            raise ValueError('Parameters are not defined for this model.')
        if len(x) != self.x_dim:
            raise TypeError(f'length of x = {len(x)} is not equal to the number of unknowns = {self.x_dim}.')

    def init_model(self):
        raise NotImplementedError()

    @property
    def param_set(self):
        return self._param_set

    @param_set.setter
    def param_set(self, param_set_processed: pd.DataFrame):
        self._param_set = param_set_processed
        self.init_model()


class LinearMixedEffectsMarginal(Base):

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
        self.bounds = Bounds(self.lb, self.ub)
        
        self.constraints = build_linear_constraint([
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

    def predict(self, x):
        betas = x[:self.n_betas]
        return np.dot(self.X, betas)


class LinearMaximal(Base):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__(param_set_processed)
    
    def objective(self, x, data: Data):
        self._prerun_check(x)
        sigma = data.obs_se
        return np.sum((data.y - np.dot(self.design_matrix, x))**2 / (2*sigma**2)) + self.prior_fun(x)

    def predict(self, x, **kwargs):
        return np.dot(self.design_matrix, x)


class FixedEffectsMaximal(LinearMaximal):

    def init_model(self):
        self.x_dim = self._param_set.num_fe
        
        self.X = self._param_set.design_matrix_fe
        self.lb = self._param_set.lb_fe
        self.ub = self._param_set.ub_fe
        self.bounds = Bounds(self.lb, self.ub)
        self.constraints = build_linear_constraint([
            (self._param_set.constr_matrix_fe, self._param_set.constr_lb_fe, self._param_set.constr_ub_fe)
        ])
        self.prior_fun = collect_priors(self._param_set.fe_priors)

    @property
    def design_matrix(self):
        return self.X


class UModel(LinearMaximal):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__(param_set_processed)
        if param_set_processed is not None:
            if not all([isinstance(prior, GaussianPrior) for prior in param_set_processed.re_priors]):
                raise TypeError('Only Gaussian priors allowed.')

    def init_model(self):
        self.x_dim = self._param_set.num_re
        self.Z = self._param_set.design_matrix_re
        self.D = self._param_set.re_var_padding
        self.lb = self._param_set.lb_re
        self.ub = self._param_set.ub_re
        self.bounds = Bounds(self.lb, self.ub)
        self.constraints = build_linear_constraint([
            (self._param_set.constr_matrix_re, self._param_set.constr_lb_re, self._param_set.constr_ub_re)
        ])
        self.gammas = [prior.std[0] for prior in self._param_set.re_priors]

    @property
    def design_matrix(self):
        return self.Z

    @property
    def prior_fun(self):
        self._prior_fun = lambda x: np.sum(x**2/np.dot(self.D, self.gammas)) / 2
        return self._prior_fun 

    def closed_form_soln(self, data: Data):
        sigma = data.obs_se 
        return np.linalg.solve(
            np.dot(self.Z.T, np.dot(np.diag(1/sigma**2), self.Z)) + np.diag(1/np.dot(self.D, self.gammas)), 
            np.dot(self.Z.T, data.y / sigma**2),
        )
    

class VModel(UModel):

    def closed_form_soln(self, data: Data):
        soln = super().closed_form_soln(data)
        return np.maximum(0, soln)
        

def build_linear_constraint(constraints: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    if len(constraints) == 1:
        mats, lbs, ubs = constraints[0]
    mats, lbs, ubs = zip(*constraints)
    A, lb, ub = combine_constraints(mats, lbs, ubs)
    if np.count_nonzero(A) == 0:
        return None 
    else:
        return LinearConstraint(A, lb, ub)
