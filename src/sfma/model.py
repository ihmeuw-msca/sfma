"""
Model class with all information to fit and predict the frontier.
"""
from operator import attrgetter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from sfma import Data, Variable, Parameter
from sfma.utils import log_erfc, dlog_erfc


class SFMAModel:

    data = property(attrgetter("_data"))
    variables = property(attrgetter("_variables"))

    def __init__(self,
                 data: Data,
                 variables: List[Variable],
                 include_ie: bool = True,
                 include_re: bool = False):
        self.include_ie = include_ie
        self.include_re = include_re
        self.data = data
        self.variables = variables

        self.opt_result = None
        self.opt_vars = None

    @data.setter
    def data(self, data: Data):
        if not isinstance(data, Data):
            raise TypeError(f"{type(self).__name__}.data must be instance of "
                            "Data.")
        self._data = data

    @variables.setter
    def variables(self, variables: List[Variable]):
        if not all(isinstance(var, Variable) for var in variables):
            raise TypeError(f"{type(self).__name__}.variables must be a list of"
                            " instances of Variable.")
        self._variables = list(variables)
        # get variable names and sizes
        self.var_names = [var.name for var in self._variables] + ["ie", "re"]
        self.var_sizes = [var.size for var in self._variables] + [1, 1]
        self.size = sum(self.var_sizes)
        # create parameter
        self.parameter = Parameter("frontier",
                                   self._variables,
                                   inv_link="identity")
        self.parameter.check_data(self._data)
        # get all variables needed for the optmization
        self.mat = self.get_mat()
        self.gvec = self.get_gprior()
        self.uvec = self.get_uprior()
        self.linear_gmat, self.linear_gvec = self.get_linear_gprior()
        self.linear_umat, self.linear_uvec = self.get_linear_uprior()

    def get_mat(self, data: Optional[Data] = None) -> np.ndarray:
        if data is None:
            data = self.data
        return self.parameter.get_mat(data)

    def get_gprior(self) -> np.ndarray:
        return np.hstack([self.parameter.get_gvec(),
                          np.array([[0.0], [np.inf]]),
                          np.array([[0.0], [np.inf]])])

    def get_uprior(self) -> np.ndarray:
        uprior = self.parameter.get_uvec()
        ie_uprior = np.array([[0.0], [np.inf]]) if self.include_ie else \
            np.array([[0.0], [0.0]])
        re_uprior = np.array([[0.0], [np.inf]]) if self.include_re else \
            np.array([[0.0], [0.0]])
        return np.hstack([uprior, ie_uprior, re_uprior])

    def get_linear_gprior(self) -> Tuple[np.ndarray, np.ndarray]:
        gmat = self.parameter.get_linear_gmat()
        gvec = self.parameter.get_linear_gvec()
        gmat = np.hstack([gmat, np.zeros((gmat.shape[0], 2))])
        return gmat, gvec

    def get_linear_uprior(self) -> Tuple[np.ndarray, np.ndarray]:
        umat = self.parameter.get_linear_umat()
        uvec = self.parameter.get_linear_uvec()
        umat = np.hstack([umat, np.zeros((umat.shape[0], 2))])
        return umat, uvec

    def _objective(self, x: np.ndarray) -> np.ndarray:
        beta, eta, gamma = x[:-2], x[-2], x[-1]

        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + gamma
        v = t + eta
        z = np.sqrt(eta)*r/np.sqrt(2*v*t)

        return 0.5*(r**2/v) + 0.5*np.log(v) - log_erfc(z)

    def objective(self, x: np.ndarray) -> float:
        return self._objective(x).dot(self.data.trim_weights)

    def _gradient(self, x: np.ndarray) -> float:
        beta, eta, gamma = x[:-2], x[-2], x[-1]

        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + gamma
        v = t + eta
        c1 = np.sqrt(eta)/np.sqrt(2*v*t)
        z = c1*r
        c2 = dlog_erfc(z)*z
        c3 = -r**2/v**2 + 1/v

        # gradient
        grad = np.zeros((x.size, self.data.num_obs), dtype=x.dtype)
        grad[:-2] = self.mat.T*(c2/r - r/v)
        grad[-2] = 0.5*(c3 + c2*(1/v - 1/eta))
        grad[-1] = 0.5*(c3 + c2*(1/v + 1/t))

        return grad

    def gradient(self, x: np.ndarray) -> float:
        return self._gradient(x).dot(self.data.trim_weights)

    def fit(self,
            x0: Optional[np.ndarray] = None,
            **options):
        x0 = np.ones(self.size) if x0 is None else x0
        bounds = self.uvec.T
        constraints = [LinearConstraint(
            self.linear_umat,
            self.linear_uvec[0],
            self.linear_uvec[1]
        )] if self.linear_uvec.size > 0 else []

        self.opt_result = minimize(self.objective, x0,
                                   method="trust-constr",
                                   jac=self.gradient,
                                   hess=self.hessian,
                                   constraints=constraints,
                                   bounds=bounds,
                                   **options)

        self.opt_vars = self.opt_result.x

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        data_pred = self.data.copy()
        data_pred.attach_df(df)
        return self.get_mat(data_pred).dot(self.opt_vars)
