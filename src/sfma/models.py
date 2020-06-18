# -*- coding: utf-8 -*-
"""
   me_model
   ~~~~~~~~

   Mixed-Effects model module.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds

from anml.models.interface import Model
from anml.parameter.parameter import ParameterSet
from anml.parameter.processors import collect_priors
from anml.parameter.variables import Variable

from sfma.data import Data


class LinearMixedEffectsMarginal(Model):

    def __init__(self, param_set_processed: ParameterSet = None):
        self.param_set = param_set_processed

    @property 
    def param_set(self):
        return self._param_set

    @param_set.setter
    def param_set(self, param_set_processed):
        self._param_set = param_set_processed
        self.n_betas = self._param_set.num_fe
        self.n_gammas = self._param_set.num_re_var
        self.x_dim = self.n_betas + self.n_gammas
        self.X = self._param_set.design_matrix
        self.Z = self._param_set.design_matrix_re
        self.C = self._param_set.constr_matrix_full
        self.lb = self._param_set.constr_lower_bounds_full
        self.ub = self._param_set.constr_upper_bounds_full
        self.bounds = Bounds(self.lb, self.ub)
        
        self.prior_fun = self._param_set.prior_fun     

    def objective(self, x, data: Data):
        if self.param_set is None:
            raise ValueError('Parameters are not yet defined for this model.')
        if len(x) != self.x_dim:
            raise TypeError(f'length of x = {len(x)} is not equal to the number of unknowns = {self.x_dim}.')
        betas = x[:self.n_betas]
        gammas = x[self.n_betas:]

        Sigma = np.diag(data.obs_se)
        Gamma = np.diag(gammas)
        y = data.obs 

        V = Sigma**2 + np.dot(self.Z,np.dot(Gamma, self.Z.T))
        r = y - self.X.dot(betas)
        return 0.5 * np.dot(r, np.linalg.solve(V, r)) + np.prod(np.linalg.slogdet(V)) + self.prior_fun(x)

    def predict(self, x):
        betas = x[:self.n_betas]
        return np.dot(self.X, betas)

    
class SingleVariableMaximal(Model):

    def __init__(self, variable_processed: Variable = None):
        self.variable = variable_processed

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, variable_processed):
        self._variable = variable_processed
        self.Z = self.variable.design_matrix_re
        self.C, self.lb, self.ub = self.variable.constraint_matrix_re()
        self.prior_fun = collect_priors([self.variable.re_prior])

        self.bounds = Bounds(self.lb, self.ub)

    def objective(self, x, data):
        if self.variable is None:
            raise ValueError('Variable is not yet defined for this model.')
        if len(x) != self.variable.n_groups:
            raise TypeError(f'The length of x = {len(x)} is not equal to the number of groups {self.variable.n_groups}.')
        y = data.obs
        sigma = data.obs_se
        return np.sum((y - x)**2 / (2*sigma**2)) + self.prior_fun(x)

    def predict(self, x):
        return np.dot(self.Z, x)






