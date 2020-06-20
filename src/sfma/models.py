# -*- coding: utf-8 -*-
"""
   me_model
   ~~~~~~~~

   Mixed-Effects model module.
"""
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint

from anml.models.interface import Model
from anml.parameter.parameter import ParameterSet, Parameter
from anml.parameter.prior import GaussianPrior, Prior
from anml.parameter.processors import process_for_marginal, process_for_maximal
from anml.parameter.variables import Variable, Intercept

from sfma.data import Data


class Base(Model):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__()
        if param_set_processed is not None:
            self.param_set = param_set_processed
        else:
            self._param_set = None

    @property
    def param_set(self):
        return self._param_set

    @param_set.setter
    def param_set(self, df: pd.DataFrame):
        raise NotImplementedError()


class LinearMixedEffectsMarginal(Base):

    @Base.param_set.setter
    def param_set(self, param_set_processed: ParameterSet):
        self._param_set = param_set_processed
        self.n_betas = self._param_set.num_fe
        self.n_gammas = self._param_set.num_re_var
        self.x_dim = self.n_betas + self.n_gammas
        
        self.X = self._param_set.design_matrix
        self.Z = self._param_set.design_matrix_re
        self.D = self._param_set.re_var_diag_matrix
        self.lb = self._param_set.lower_bounds_full 
        self.ub = self._param_set.upper_bounds_full  
        self.bounds = Bounds(self.lb, self.ub)
        if self._param_set.constr_matrix_full is not None:      
            self.C = self._param_set.constr_matrix_full
            self.c_lb = self._param_set.constr_lower_bounds_full
            self.c_ub = self._param_set.constr_upper_bounds_full
            self.constraints = LinearConstraint(self.C, self.c_lb, self.c_ub)
        self.prior_fun = self._param_set.prior_fun 

    def objective(self, x, data: Data):
        if self._param_set is None:
            raise ValueError('Parameters are not defined for this model.')
        if len(x) != self.x_dim:
            raise TypeError(f'length of x = {len(x)} is not equal to the number of unknowns = {self.x_dim}.')
        betas = x[:self.n_betas]
        gammas = x[self.n_betas:]

        Sigma = np.diag(data.obs_se)
        Gamma = np.diag(np.dot(self.D, gammas))
        y = data.obs 

        V = Sigma**2 + np.dot(self.Z,np.dot(Gamma, self.Z.T))
        r = y - self.X.dot(betas)
        return 0.5 * np.dot(r, np.linalg.solve(V, r)) + np.prod(np.linalg.slogdet(V)) + self.prior_fun(x)

    def predict(self, x):
        betas = x[:self.n_betas]
        return np.dot(self.X, betas)


class LinearMaximal(Base):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__(param_set_processed)
    
    def objective(self, x, data: Data):
        if self.param_set is None:
            raise ValueError('Parameter set is not yet defined for this model.')
        if len(x) != self.param_set.num_re:
            raise TypeError(f'The length of x = {len(x)} is not equal to the number of variables {self.x_dim}.')
        y = data.obs
        sigma = data.obs_se
        return np.sum((y - np.dot(self.design_matrix, x))**2 / (2*sigma**2)) + self.prior_fun(x)

    def predict(self, x, **kwargs):
        return np.dot(self.design_matrix, x)


class FixedEffectsMaximal(LinearMaximal):

    @LinearMaximal.param_set.setter 
    def param_set(self, param_set_processed):
        self._param_set = param_set_processed
        self.n_betas = self._param_set.num_fe
        self.x_dim = self.n_betas
        
        self.design_matrix = self._param_set.design_matrix
        self.lb = self._param_set.lower_bounds_full 
        self.ub = self._param_set.upper_bounds_full  
        self.bounds = Bounds(self.lb, self.ub)
        if self._param_set.constr_matrix_full is not None:      
            self.C = self._param_set.constr_matrix_full
            self.c_lb = self._param_set.constr_lower_bounds_full
            self.c_ub = self._param_set.constr_upper_bounds_full
            self.constraints = LinearConstraint(self.C, self.c_lb, self.c_ub)
        self.prior_fun = self._param_set.prior_fun 


class UModel(LinearMaximal):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__(param_set_processed)
        if param_set_processed is not None:
            if not all([isinstance(prior, GaussianPrior) for prior in param_set_processed.re_priors]):
                raise TypeError('Only Gaussian priors allowed.')

    @LinearMaximal.param_set.setter
    def param_set(self, param_set_processed: pd.DataFrame):
        self._param_set = param_set_processed
        self.design_matrix = self._param_set.design_matrix_re
        self.D = self._param_set.re_var_diag_matrix
        self.lb = self._param_set.lower_bounds_full[self._param_set.num_fe:]
        self.ub = self._param_set.upper_bounds_full[self._param_set.num_fe:]
        self.bounds = Bounds(self.lb, self.ub)
        if self._param_set.constr_matrix_full is not None:
            self.C = self._param_set.constr_matrix_full[:, self._param_set.num_fe:]
            self.c_lb = self._param_set.constr_lower_bounds_full
            self.c_ub = self._param_set.constr_upper_bounds_full
            self.constraints = LinearConstraint(self.C, self.c_lb, self.c_ub)
        self.x_dim = self._param_set.num_re
        self.gammas = [prior.std[0] for prior in param_set_processed.re_priors]

    @property
    def prior_fun(self):
        self._prior_fun = lambda x: np.sum(x**2/np.dot(self.D, self.gammas)) / 2
        return self._prior_fun 

    def closed_form_soln(self, data: Data):
        y = data.obs
        sigma = data.obs_se 
        return np.linalg.solve(
            np.dot(self.design_matrix.T, np.dot(np.diag(1/sigma**2), self.design_matrix)) + np.diag(1/np.dot(self.D, self.gammas)), 
            np.dot(self.design_matrix.T, y / sigma**2),
        )
    

class VModel(UModel):

    def closed_form_soln(self, data: Data):
        soln = super().closed_form_soln(data)
        return np.maximum(0, soln)
        

