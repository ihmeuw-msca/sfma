"""
Model class with all information to fit and predict the frontier.
"""
from operator import attrgetter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from anml.parameter.main import Parameter
from anml.variable.main import Variable
from msca.c2fun import logerfc
from msca.linalg.matrix import asmatrix
from msca.optim.prox import proj_capped_simplex
from msca.optim.solver import IPSolver, NTSolver
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.optimize import minimize_scalar

from sfma.data import Data


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

    """

    data = property(attrgetter("_data"))
    variables = property(attrgetter("_variables"))

    def __init__(self,
                 data: Data,
                 variables: List[Variable],
                 include_ie: bool = True,
                 include_re: bool = False,
                 df: Optional[DataFrame] = None):
        self.data = data
        self.parameter = Parameter(variables)
        self.include_ie = include_ie
        self.include_re = include_re

        # get all variables needed for the optmization
        self.mat = None
        self.cmat = None
        self.cvec = None
        self.weights = None

        # initialize the variables
        self.beta = np.ones(self.parameter.size)
        self.eta = float(include_ie)
        self.gamma = float(include_re)

        if df is not None:
            self.attach(df)

    @data.setter
    def data(self, data: Data):
        if not isinstance(data, Data):
            raise TypeError(f"{type(self).__name__}.data must be instance of "
                            "Data.")
        self._data = data

    def get_beta_dict(self) -> Dict[str, NDArray]:
        sizes = [variable.size for variable in self.parameter.variables]
        betas = np.split(self.beta, np.cumsum(sizes)[:-1])
        return {
            variable.component.key: betas[i]
            for i, variable in enumerate(self.parameter.variables)
        }

    def attach(self, df: DataFrame):
        self.data.attach(df)
        self.parameter.attach(df)
        if self.mat is None:
            self.mat = asmatrix(self.parameter.design_mat)
        if self.weights is None:
            self.weights = np.ones(df.shape[0])
        if (self.cmat is None) or (self.cvec is None):
            prior = self.parameter.prior_dict["linear"]["UniformPrior"]
            mat, vec = prior.mat, prior.params
            prior = self.parameter.prior_dict["direct"]["UniformPrior"]
            cmat = np.vstack([mat, np.identity(self.parameter.size)])
            cvec = np.hstack([vec, prior.params])

            index = ~np.isclose(cmat, 0.0).all(axis=1)
            cmat = cmat[index]
            cvec = cvec[:, index]

            scale = np.abs(cmat).max(axis=1)
            cmat = cmat / scale[:, np.newaxis]
            cvec = cvec / scale

            self.cmat = asmatrix(np.vstack([
                -cmat[~np.isneginf(cvec[0])], cmat[~np.isposinf(cvec[1])]
            ]))
            self.cvec = np.hstack([
                -cvec[0][~np.isneginf(cvec[0])], cvec[1][~np.isposinf(cvec[1])]
            ])

    def _objective(self,
                   beta: Optional[NDArray] = None,
                   eta: Optional[float] = None,
                   gamma: Optional[float] = None) -> NDArray:
        """Objective function for each data point.

        Parameters
        ----------
        beta : Optional[NDArray], optional
            Beta variable, by default None. When it is None, use `self.beta`.
        eta : Optional[float], optional
            Eta variable, by default None. When it is None, use `self.eta`.
        gamma : Optional[float], optional
            Gamma variable, by default None. When it is None, use `self.gamma`.

        Returns
        -------
        NDArray
            Objective value for each data point.
        """
        beta = self.beta if beta is None else beta
        eta = self.eta if eta is None else eta
        gamma = self.gamma if gamma is None else gamma

        r = self.data.obs.value - self.mat.dot(beta)
        t = self.data.obs_se.value**2 + gamma
        v = t + eta
        z = np.sqrt(eta)/np.sqrt(2*v*t)

        return 0.5*(r**2/v) + 0.5*np.log(2*np.pi*v) - logerfc(z*r)

    def objective_beta(self, beta: NDArray) -> float:
        """Objective value with respect to beta.

        Parameters
        ----------
        beta : NDArray
            Beta variable.

        Returns
        -------
        float
            Objective value.
        """
        return self.weights.dot(self._objective(beta=beta)) + \
            self.parameter.prior_objective(beta)

    def gradient_beta(self, beta: NDArray) -> NDArray:
        """Gradient vector with respect to beta.

        Parameters
        ----------
        beta : NDArray
            Beta variable.

        Returns
        -------
        NDArray
            Gradient vector.
        """
        r = self.data.obs.value - self.mat.dot(beta)
        t = self.data.obs_se.value**2 + self.gamma
        v = t + self.eta
        z = np.sqrt(self.eta)/np.sqrt(2*v*t)

        dlr = -r/v
        dzr = logerfc(z*r, order=1)

        return self.mat.T.dot(self.weights*(dlr + dzr*z)) + \
            self.parameter.prior_gradient(beta)

    def hessian_beta(self, beta: NDArray) -> NDArray:
        """Hessian matrix with respect to beta.

        Parameters
        ----------
        beta : NDArray
            Beta variable.

        Returns
        -------
        NDArray
            Hessian matrix.
        """
        w = self.weights

        r = self.data.obs.value - self.mat.dot(beta)
        t = self.data.obs_se.value**2 + self.gamma
        v = t + self.eta
        z = np.sqrt(self.eta)/np.sqrt(2*v*t)

        d2lr = 1/v
        d2zr = logerfc(z*r, order=2)

        return (self.mat.T*(w*(d2lr - d2zr*z**2))).dot(self.mat) + \
            self.parameter.prior_hessian(beta)

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
        return self.weights.dot(self._objective(eta=eta))

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
        r = self.data.obs.value - self.mat.dot(self.beta)
        t = self.data.obs_se.value**2 + self.gamma
        v = t + eta
        z = np.sqrt(eta)/np.sqrt(2*v*t)

        dzr = logerfc(z*r, order=1)
        dlv = 0.5*(-r**2/v**2 + 1/v)
        dze = 0.5*z*(1/eta - 1/v)

        return self.weights.dot(dlv - dzr*r*dze)

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
        return self.weights.dot(self._objective(gamma=gamma))

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
        r = self.data.obs.value - self.mat.dot(self.beta)
        t = self.data.obs_se.value**2 + gamma
        v = t + self.eta
        z = np.sqrt(self.eta)/np.sqrt(2*v*t)

        dzr = logerfc(z*r, order=1)
        dlv = 0.5*(-r**2/v**2 + 1/v)
        dzg = 0.5*z*(-1/t - 1/v)

        return self.weights.dot(dlv - dzr*r*dzg)

    def _fit_beta(self,
                  beta0: Optional[NDArray] = None,
                  **options):
        """Partially minimize beta.

        Parameters
        ----------
        beta0 : Optional[NDArray], optional
            Initial guess of beta variable, by default None. When it is None,
            use `self.beta`.
        solver_type: {'ip', 'pg', 'sp'}, optional
            Solver type, 'ip' stands for interior point solver and 'pg' stands
            for projected gradient solver.
        """
        beta0 = beta0 or self.beta.copy()

        if self.cmat.size == 0:
            solver = NTSolver(
                self.objective_beta,
                self.gradient_beta,
                self.hessian_beta,
            )
        else:
            solver = IPSolver(
                self.objective_beta,
                self.gradient_beta,
                self.hessian_beta,
                self.cmat,
                self.cvec
            )
        result = solver.minimize(beta0, **options)
        self.beta = result.x

    def _fit_eta(self, **options):
        """Paratial minimize eta."""
        options = {"method": "bounded", "bounds": [0.0, 1.0], **options}
        if self.include_ie:
            result = minimize_scalar(self.objective_eta,
                                     **options)
            self.eta = result.x

    def _fit_gamma(self, **options):
        """Partial minimize gamma"""
        options = {"method": "bounded", "bounds": [0.0, 1.0], **options}
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
                  f"eta={self.eta:.2e}, gamma={self.gamma:.2e}, "
                  f"error={error:.2e}")

        while error >= tol and counter < max_iter:
            counter += 1

            self._fit_beta(**beta_options)
            self._fit_eta(**eta_options)
            self._fit_gamma(**gamma_options)

            x_new = np.hstack([self.beta, self.eta, self.gamma])
            error = np.max(np.abs(x_new - x))
            x = x_new

            if verbose:
                print(f"{counter=:3d}, "
                      f"obj={self.objective_beta(self.beta):.2e}, "
                      f"eta={self.eta:.2e}, gamma={self.gamma:.2e} "
                      f"error={error:.2e}")

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
        outlier_pct = max(0.0, min(1.0, outlier_pct))
        inlier_pct = 1 - outlier_pct
        if 0.0 < outlier_pct < 1.0:
            num_obs = self.data.obs.value.size
            num_inliers = int(num_obs*inlier_pct)
            self.weights = np.full(num_obs, inlier_pct)

            w = self.weights.copy()
            trim_error = 1.0
            trim_counter = 0

            if trim_verbose:
                print(f"{trim_counter=:3d}, "
                      f"obj={self.objective_beta(self.beta):.2e}, "
                      f"{trim_error=:.2e}")

            while trim_error >= trim_tol and trim_counter < trim_max_iter:
                trim_counter += 1

                trim_grad = self._objective()
                self.weights = proj_capped_simplex(
                    w - trim_step_size*trim_grad, num_inliers
                )

                trim_error = np.linalg.norm(w - self.weights)
                w = self.weights.copy()

                if trim_verbose:
                    print(f"{trim_counter=:3d}, "
                          f"obj={self.objective_beta(self.beta):.2e}, "
                          f"{trim_error=:.2e}")

                self._fit(**options)
        else:
            self.weights = np.ones(self.data.obs.value.size)
            self._fit(**options)

    def get_inefficiency(self) -> NDArray:
        """Estimate inefficiency.

        Returns
        -------
        NDArray
            Array of inefficiency for given data.
        """
        r = self.data.obs.value - self.mat.dot(self.beta)
        return np.maximum(
            0.0, -self.eta * r / (self.data.obs_se.value**2 +
                                  self.eta + self.gamma)
        )

    def predict(self, df: pd.DataFrame) -> NDArray:
        """Model predicting function.

        Parameters
        ----------
        df : pd.DataFrame
            Input prediction data frame.

        Returns
        -------
        NDArray
            Prediction of the frontier.
        """
        return self.parameter.get_params(self.beta, df=df)
