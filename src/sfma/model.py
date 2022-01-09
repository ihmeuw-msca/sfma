"""
Model class with all information to fit and predict the frontier.
"""
from operator import attrgetter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize_scalar

from sfma import Data, Parameter, Variable
from sfma.solver import IPSolver, PGSolver, SPSolver, proj_csimplex
from sfma.solver.projector import PolyProjector
from sfma.utils import d2log_erfc, dlog_erfc, log_erfc


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
    parameter : Parameter
        Parameter instance to group all variables for convenient design matrices
        building.
    mat : np.ndarray
        Design matrix.
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
    objective_beta(beta)
        Objective value with respect to beta.
    gradient_beta(beta)
        Gradient vector with respect to beta.
    hessian_beta(beta)
        Hessian matrix with respect to beta.
    objective_eta(eta)
        Objective value with respect to eta.
    gradient_eta(eta)
        Gradient value with respect to eta.
    objective_gamma(gamma)
        Objective value with respect to gamma.
    gradient_gamma(gamma)
        Gradient value with respect to gamma.
    fit(outlier_pct, trim_max_iter, trim_step_size, trim_tol, trim_verbose,
        **options)
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

        # create parameter
        self.parameter = Parameter("frontier", self.variables, inv_link="identity")
        self.parameter.check_data(self.data)

        # get all variables needed for the optmization
        self.mat = self.get_mat()
        self.linear_gmat, self.linear_gvec = self.get_linear_gprior()
        self.linear_umat, self.linear_uvec = self.get_linear_uprior()

        # initialize the variables
        self.beta = np.ones(self.parameter.size)
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

    def get_linear_gprior(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get linear Gaussian prior.

        Returns
        -------
        np.ndarray
            Linear Gaussian prior.
        """
        gmat = self.parameter.get_linear_gmat()
        gvec = self.parameter.get_linear_gvec()
        gmat = np.vstack([gmat, np.identity(self.parameter.size)])
        gvec = np.hstack([gvec, self.parameter.get_gvec()])
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
        umat = np.vstack([umat, np.identity(self.parameter.size)])
        uvec = np.hstack([uvec, self.parameter.get_uvec()])
        return umat, uvec

    def _objective(self,
                   beta: Optional[np.ndarray] = None,
                   eta: Optional[float] = None,
                   gamma: Optional[float] = None) -> np.ndarray:
        """Objective function for each data point.

        Parameters
        ----------
        beta : Optional[np.ndarray], optional
            Beta variable, by default None. When it is None, use `self.beta`.
        eta : Optional[float], optional
            Eta variable, by default None. When it is None, use `self.eta`.
        gamma : Optional[float], optional
            Gamma variable, by default None. When it is None, use `self.gamma`.

        Returns
        -------
        np.ndarray
            Objective value for each data point.
        """
        beta = self.beta if beta is None else beta
        eta = self.eta if eta is None else eta
        gamma = self.gamma if gamma is None else gamma

        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + gamma
        v = t + eta
        z = np.sqrt(eta)/np.sqrt(2*v*t)

        return 0.5*(r**2/v) + 0.5*np.log(2*np.pi*v) - log_erfc(z*r)

    def objective_beta(self, beta: np.ndarray) -> float:
        """Objective value with respect to beta.

        Parameters
        ----------
        beta : np.ndarray
            Beta variable.

        Returns
        -------
        float
            Objective value.
        """
        prior_r = self.linear_gmat.dot(beta) - self.linear_gvec[0]
        return self.data.trim_weights.dot(self._objective(beta=beta)) + \
            0.5*np.sum(prior_r**2/self.linear_gvec[1]**2)

    def gradient_beta(self, beta: np.ndarray) -> np.ndarray:
        """Gradient vector with respect to beta.

        Parameters
        ----------
        beta : np.ndarray
            Beta variable.

        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + self.gamma
        v = t + self.eta
        z = np.sqrt(self.eta)/np.sqrt(2*v*t)

        dlr = -r/v
        dzr = dlog_erfc(z*r)

        prior_r = self.linear_gmat.dot(beta) - self.linear_gvec[0]
        return self.mat.T.dot(self.data.trim_weights*(dlr + dzr*z)) + \
            self.linear_gmat.T.dot(prior_r / self.linear_gvec[1]**2)

    def hessian_beta(self, beta: np.ndarray) -> np.ndarray:
        """Hessian matrix with respect to beta.

        Parameters
        ----------
        beta : np.ndarray
            Beta variable.

        Returns
        -------
        np.ndarray
            Hessian matrix.
        """
        w = self.data.trim_weights

        r = self.data.obs - self.mat.dot(beta)
        t = 1/self.data.weights + self.gamma
        v = t + self.eta
        z = np.sqrt(self.eta)/np.sqrt(2*v*t)

        d2lr = 1/v
        d2zr = d2log_erfc(z*r)

        return (self.mat.T*(w*(d2lr - d2zr*z**2))).dot(self.mat) + \
            (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat)

    def objective_eta(self, eta: float) -> float:
        """Objective value with respect to eta.

        Parameters
        ----------
        eta : float
            Eta variable.

        Returns
        -------
        float
            Objective value.
        """
        return self.data.trim_weights.dot(self._objective(eta=eta))

    def gradient_eta(self, eta: float) -> float:
        """Gradient value with repsect to eta.

        Parameters
        ----------
        eta : float
            Eta variable.

        Returns
        -------
        float
            Derivative of eta.
        """
        r = self.data.obs - self.mat.dot(self.beta)
        t = 1/self.data.weights + self.gamma
        v = t + eta
        z = np.sqrt(eta)/np.sqrt(2*v*t)

        dzr = dlog_erfc(z*r)
        dlv = 0.5*(-r**2/v**2 + 1/v)
        dze = 0.5*z*(1/eta - 1/v)

        return self.data.trim_weights.dot(dlv - dzr*r*dze)

    def objective_gamma(self, gamma: float) -> float:
        """Objective value with respect to gamma.

        Parameters
        ----------
        gamma : float
            Gamma variable.

        Returns
        -------
        float
            Objective value.
        """
        return self.data.trim_weights.dot(self._objective(gamma=gamma))

    def gradient_gamma(self, gamma: float) -> float:
        """Gradient value with respect to gamma.

        Parameters
        ----------
        gamma : float
            Gamma variable.

        Returns
        -------
        float
            Derivative of gamma.
        """
        r = self.data.obs - self.mat.dot(self.beta)
        t = 1/self.data.weights + gamma
        v = t + self.eta
        z = np.sqrt(self.eta)/np.sqrt(2*v*t)

        dzr = dlog_erfc(z*r)
        dlv = 0.5*(-r**2/v**2 + 1/v)
        dzg = 0.5*z*(-1/t - 1/v)

        return self.data.trim_weights.dot(dlv - dzr*r*dzg)

    def _fit_beta(self,
                  beta0: Optional[np.ndarray] = None,
                  solver_type: str = "ip",
                  **options):
        """Partially minimize beta.

        Parameters
        ----------
        beta0 : Optional[np.ndarray], optional
            Initial guess of beta variable, by default None. When it is None,
            use `self.beta`.
        solver_type: {'ip', 'pg', 'sp'}, optional
            Solver type, 'ip' stands for interior point solver and 'pg' stands
            for projected gradient solver.
        """
        beta0 = self.beta.copy() if beta0 is None else beta0
        constraint = LinearConstraint(
            self.linear_umat,
            self.linear_uvec[0],
            self.linear_uvec[1]
        )

        if solver_type == "ip":
            solver = IPSolver(self.gradient_beta,
                              self.hessian_beta,
                              constraint)
        elif solver_type == "pg":
            projector = PolyProjector(constraint)
            solver = PGSolver(self.objective_beta,
                              self.gradient_beta,
                              self.hessian_beta,
                              projector)
        elif solver_type == "sp":
            solver = SPSolver(self.objective_beta,
                              self.gradient_beta,
                              self.hessian_beta,
                              [constraint])
        else:
            raise ValueError("Unrecognized solver type, must be 'ip' or 'pg'.")
        self.beta = solver.minimize(beta0, **options)

    def _fit_eta(self, **options):
        """Paratial minimize eta."""
        if self.include_ie:
            result = minimize_scalar(self.objective_eta,
                                     **options)
            self.eta = result.x

    def _fit_gamma(self, **options):
        """Partial minimize gamma"""
        if self.include_re:
            result = minimize_scalar(self.objective_gamma,
                                     **options)
            self.gamma = result.x

    def _fit(self,
             max_iter: int = 20,
             tol: float = 1e-2,
             verbose: bool = False,
             beta_options: Optional[Dict] = None,
             eta_options: Optional[Dict] = None,
             gamma_options: Optional[Dict] = None):
        """Model fitting function.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iteration, by default 20.
        tol : float, optional
            Tolerance of convergence, by default 1e-2.
        verbose : bool, optional
            If True print out convergence information, by default False.
        beta_options : Optional[Dict], optional
            Optimizer options for beta, by default None.
        eta_options : Optional[Dict], optional
            Optimizer options for eta, by default None.
        gamma_options : Optional[Dict], optional
            Optimizer options for gamma, by default None.
        """
        counter = 0
        error = 1.0

        beta_options = {} if beta_options is None else beta_options
        eta_options = {} if eta_options is None else eta_options
        gamma_options = {} if gamma_options is None else gamma_options

        x = np.hstack([self.beta, self.eta, self.gamma])
        if verbose:
            print(f"{counter=:3d}, obj={self.objective_beta(self.beta):.2e}, "
                  f"eta={self.eta:.2e}, gamma={self.gamma:.2e}")

        while error >= tol and counter < max_iter:
            counter += 1

            self._fit_beta(**beta_options)
            self._fit_eta(**eta_options)
            self._fit_gamma(**gamma_options)

            x_new = np.hstack([self.beta, self.eta, self.gamma])
            error = np.max(np.abs(x_new - x))

            if verbose:
                print(f"{counter=:3d}, "
                      f"obj={self.objective_beta(self.beta):.2e}, "
                      f"eta={self.eta:.2e}, gamma={self.gamma:.2e}")

    def fit(self,
            outlier_pct: float = 0.0,
            trim_max_iter: int = 5,
            trim_step_size: float = 1.0,
            trim_tol: float = 1e-5,
            trim_verbose: bool = False,
            **options):
        """Model fitting function.

        Parameters
        ----------
        outlier_pct : float, optional
            Outlier percentage, by default 0.0.
        trim_max_iter : int, optional
            Maximum trimming iteration, by default 5.
        trim_step_size : float, optional
            Trimming step size, by default 1.0.
        trim_tol : float, optional
            Trimming tolerance, by default 1e-5.
        trim_verbose : bool, optional
            If True, print out trimming convergence information, by default
            False.
        """
        self._fit(**options)
        outlier_pct = max(0.0, min(1.0, outlier_pct))
        inlier_pct = 1 - outlier_pct
        if 0.0 < outlier_pct < 1.0:
            num_inliers = int(self.data.num_obs*inlier_pct)
            self.data.trim_weights = np.full(self.data.df.shape[0], inlier_pct)

            w = self.data.trim_weights.copy()
            trim_error = 1.0
            trim_counter = 0

            if trim_verbose:
                print(f"{trim_counter=:3d}, "
                      f"obj={self.objective_beta(self.beta):.2e}, "
                      f"{trim_error=:.2e}")

            while trim_error >= trim_tol and trim_counter < trim_max_iter:
                trim_counter += 1

                trim_grad = self._objective()
                self.data.trim_weights = proj_csimplex(
                    w - trim_step_size*trim_grad, num_inliers
                )

                trim_error = np.linalg.norm(w - self.data.trim_weights)
                w = self.data.trim_weights.copy()

                if trim_verbose:
                    print(f"{trim_counter=:3d}, "
                          f"obj={self.objective_beta(self.beta):.2e}, "
                          f"{trim_error=:.2e}")

                self._fit(**options)

    def get_inefficiency(self) -> np.ndarray:
        """Estimate inefficiency.

        Returns
        -------
        np.ndarray
            Array of inefficiency for given data.
        """
        r = self.data.obs - self.mat.dot(self.beta)
        return np.maximum(
            0.0, -self.eta * r / (1 / self.data.weights + self.eta + self.gamma)
        )

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
        data_pred = self.data.copy()
        data_pred.attach_df(df)
        return self.get_mat(data_pred).dot(self.beta)
