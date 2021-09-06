"""
Model class with all information to fit and predict the frontier.
"""
from operator import attrgetter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from sfma import Data, Variable, Parameter
from sfma.utils import log_erfc, dlog_erfc, d2log_erfc


class SFMAModel:
    """SFMA model class with all information for fitting and predicting.

    Parameters
    ----------
    data : Data
        Data object with observations and covariates.
    variables : List[Variable]
        List of variables model the frontier.
    include_ie : bool, optional
        If `True` includes inefficiency into the model. Default to be True.
    include_re : bool, optional
        If `True` includes random effects into the model. Default to be False.

    Attributes
    ----------
    data : Data
        Data object with observations and covariates.
    variables : List[Variable]
        List of variables model the frontier.
    include_ie : bool
        If `True` includes inefficiency into the model.
    include_re : bool
        If `True` includes random effects into the model.
    var_names : List[str]
        List of variable names.
    var_sizes : List[int]
        List of variable sizes.
    size : int
        Total size of variables
    parameter : Parameter
        Parameter instance to group all variables for convenient design matrices
        building.
    mat : np.ndarray
        Design matrix.
    gvec : np.ndarray
        Direct Gaussian prior information.
    uvec : np.ndarray
        Direct Uniform prior information.
    linear_gmat : np.ndarray
        Linear Gaussian prior mapping.
    linear_gvec : np.ndarray
        Linear Guassian prior information.
    linear_umat : np.ndarray
        Linear Uniform prior mapping.
    linear_uvec : np.ndarray
        Linear Uniform prior information.

    Methods
    -------
    get_mat(data)
        Get design matrix.
    get_grior()
        Get direct Gaussian prior.
    get_uprior()
        Get direct Uniform prior.
    get_linear_gprior()
        Get linear Gaussian prior.
    get_linear_uprior()
        Get linear Uniform prior.
    get_vars(x)
        Process vector into variables.
    objective(x)
        Objective function.
    gradient(x)
        Gradient function.
    hessian(x)
        Hessian function.
    fit(x0, outlier_pct, num_steps, **options)
        Model fitting function.
    predict(df)
        Model predicting function.
    """

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
        self.beta = np.ones(self.size - 2)
        self.eta = float(include_ie)
        self.gamma = float(include_re)

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
        """Get design matrix.

        Parameters
        ----------
        data : Optional[Data], optional
            Data used to generate design matrix, by default None. If None, it
            will use the `self.data` as the data object.

        Returns
        -------
        np.ndarray
            Design matrix.
        """
        if data is None:
            data = self.data
        return self.parameter.get_mat(data)

    def get_gprior(self) -> np.ndarray:
        """Get direct Gaussian prior.

        Returns
        -------
        np.ndarray
            Direct Gaussian prior.
        """
        return self.parameter.get_gvec()

    def get_uprior(self) -> np.ndarray:
        """Get direct Uniform prior.

        Returns
        -------
        np.ndarray
            Direct Uniform prior.
        """
        uprior = self.parameter.get_uvec()
        ie_uprior = np.array([[0.0], [np.inf]]) if self.include_ie else \
            np.array([[0.0], [0.0]])
        re_uprior = np.array([[0.0], [np.inf]]) if self.include_re else \
            np.array([[0.0], [0.0]])
        return np.hstack([uprior, ie_uprior, re_uprior])

    def get_linear_gprior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linear Gaussian prior.

        Returns
        -------
        np.ndarray
            Linear Gaussian prior.
        """
        gmat = self.parameter.get_linear_gmat()
        gvec = self.parameter.get_linear_gvec()
        return gmat, gvec

    def get_linear_uprior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linear Uniform prior.

        Returns
        -------
        np.ndarray
            Linear Uniform prior.
        """
        umat = self.parameter.get_linear_umat()
        uvec = self.parameter.get_linear_uvec()
        umat = np.hstack([umat, np.zeros((umat.shape[0], 2))])
        return umat, uvec

    def get_vars(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Process vector into variables.

        Parameters
        ----------
        x : np.ndarray
            Input vector.

        Returns
        -------
        Tuple[np.ndarray, float, float]
            Returns beta, eta and gamma.
        """
        beta, eta, gamma = x[:-2], np.sqrt(x[-2]**2), np.sqrt(x[-1]**2)
        eta = float(self.include_ie)*eta
        gamma = float(self.include_re)*gamma
        return beta, eta, gamma

    def _objective(self, x: np.ndarray) -> float:
        beta, eta, gamma = self.get_vars(x)

        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + gamma
        v = t + eta
        z = np.sqrt(eta)/np.sqrt(2*v*t)

        return 0.5*(r**2/v) + 0.5*np.log(v) - log_erfc(z*r)

    def objective(self, x: np.ndarray) -> float:
        """Objective function.

        Parameters
        ----------
        x : np.ndarray
            Input variable.

        Returns
        -------
        float
            Objective value.
        """
        beta, _, _ = self.get_vars(x)
        w = self.data.trim_weights

        value = w.dot(self._objective(x))
        value += 0.5*np.sum(((beta - self.gvec[0])/self.gvec[1])**2)
        value += 0.5*np.sum(
            ((self.linear_gmat.dot(beta) - self.linear_gvec[0]) /
             self.linear_gvec[1])**2
        )
        return value

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient function.

        Parameters
        ----------
        x : np.ndarray
            Input variable.

        Returns
        -------
        float
            Gradient vector.
        """
        beta, eta, gamma = self.get_vars(x)
        w = self.data.trim_weights

        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + gamma
        v = t + eta
        z = np.sqrt(eta)/np.sqrt(2*v*t)

        dlr = -r/v
        dzr = dlog_erfc(z*r)
        dlv = 0.5*(-r**2/v**2 + 1/v)
        dze = 0.5*z*(1/eta - 1/v)
        dzg = 0.5*z*(-1/t - 1/v)

        # gradient
        grad_r = self.mat.T.dot(w*(dlr + dzr*z))
        grad_e = w.dot(dlv - dzr*r*dze)
        grad_g = w.dot(dlv - dzr*r*dzg)

        grad_r += (beta - self.gvec[0])/self.gvec[1]**2
        grad_r += self.linear_gmat.T.dot(
            (self.linear_gmat.dot(beta) - self.linear_gvec[0]) /
            self.linear_gvec[1]
        )
        return np.hstack([grad_r, grad_e, grad_g])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Hessian function.

        Parameters
        ----------
        x : np.ndarray
            Input variable.

        Returns
        -------
        np.ndarray
            Hessian matrix.
        """
        beta, eta, gamma = self.get_vars(x)
        w = self.data.trim_weights

        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + gamma
        v = t + eta
        z = np.sqrt(eta)/np.sqrt(2*v*t)

        dzr = dlog_erfc(z*r)
        dze = 0.5*z*(1/eta - 1/v)
        dzg = 0.5*z*(-1/t - 1/v)

        d2lr = 1/v
        dlrv = r/v**2
        d2zr = d2log_erfc(z*r)
        d2lv = r**2/v**3 - 0.5/v**2
        d2ze = 0.25*z*(1/eta - 1/v)**2 + 0.5*z*(-1/eta**2 + 1/v**2)
        dzeg = 0.25*z*(1/eta - 1/v)*(-1/t - 1/v) + 0.5*z/v**2
        d2zg = 0.25*z*(1/t + 1/v)**2 + 0.5*z*(1/t**2 + 1/v**2)

        hess_rr = (self.mat.T*(w*(d2lr - d2zr*z**2))).dot(self.mat)
        hess_re = self.mat.T.dot(w*(dlrv + d2zr*(z*r)*dze + dzr*dze))
        hess_rg = self.mat.T.dot(w*(dlrv + d2zr*(z*r)*dzg + dzr*dzg))
        hess_ee = w.dot(d2lv - d2zr*(r**2)*(dze**2) - dzr*r*d2ze)
        hess_eg = w.dot(d2lv - d2zr*(r**2)*(dze*dzg) - dzr*r*dzeg)
        hess_gg = w.dot(d2lv - d2zr*(r**2)*(dzg**2) - dzr*r*d2zg)

        hess_rr += np.diag(1/self.gvec[1]**2) + \
            (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat)

        return np.block([
            [hess_rr, hess_re[:, None], hess_rg[:, None]],
            [hess_re, hess_ee, hess_eg],
            [hess_rg, hess_eg, hess_gg]
        ])

    def _fit(self,
             x0: Optional[np.ndarray] = None,
             **options):
        """Model fitting function.

        Parameters
        ----------
        x0 : Optional[np.ndarray], optional
            Initial input variable, by default None. If None, it will use all
            one vector as the initial guess.
        """
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
        self.beta = self.opt_result.x[:-2]
        self.eta = self.opt_result.x[-2]
        self.gamma = self.opt_result.x[-1]

    def fit(self,
            x0: Optional[np.ndarray] = None,
            outlier_pct: float = 0.0,
            num_steps: int = 5,
            **options):
        """Model fitting function.

        Parameters
        ----------
        x0 : Optional[np.ndarray], optional
            Initial input variable, by default None. If None, it will use all
            one vector as the initial guess.
        outlier_pct : float, optional
            Outlier percentage. Default to be 0.
        num_steps : int, optional
            Number of trimming steps. Default to be 5.
        """
        self._fit(x0, **options)
        outlier_pct = max(0.0, min(1.0, outlier_pct))
        if 0.0 < outlier_pct < 1.0:
            num_outliers = int(self.data.num_obs*outlier_pct)
            weights = np.linspace(1.0, 0.0, num_steps)
            for weight in weights[1:]:
                # update trimming weights
                indices = np.argsort(self._objective(self.opt_vars))[::-1]
                indices = indices[:num_outliers]
                self.data.df["trim_weights"] = 1.0
                self.data.df.loc[indices, "trim_weights"] = weight
                self._fit(x0=self.opt_vars, **options)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Model predicting function.

        Parameters
        ----------
        df : pd.DataFrame
            Input prediction data frame.

        Returns
        -------
        np.ndarray
            Prediction of the frontier.
        """
        beta, _, _ = self.get_vars(self.opt_vars)
        data_pred = self.data.copy()
        data_pred.attach_df(df)
        return self.get_mat(data_pred).dot(beta)
