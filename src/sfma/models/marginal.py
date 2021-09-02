from typing import Tuple, List, Optional
import numpy as np
from numpy import ndarray

from anml.models.interface import TrimmingCompatibleModel
from anml.parameter.parameter import Parameter

from sfma.data import Data
from sfma.models.utils import build_linear_constraint, log_erfc
from scipy.special import erfc


class MarginalModel(TrimmingCompatibleModel):
    """Marginal model for stochastic frontier.
    """

    def __init__(self, params: List[Parameter]):
        super().__init__()
        self._w = None

        if not all([isinstance(param, Parameter) for param in params]):
            raise TypeError("params must be a list of Parameter.")
        param_names = [param.param_name for param in params]
        if "eta" not in param_names:
            raise ValueError("MarginalModel requires parameter eta.")
        if "gamma" not in param_names:
            raise ValueError("MarginalModel requires parameter gamma.")
        if not any(["beta" in x for x in param_names]):
            raise ValueError("MarginalModel requires parameter beta.")
        self.params = {
            param.param_name: param
            for param in params
        }

        # extract constraints information
        self.lb = np.hstack([self.params[name].lb_fe for name in param_names])
        self.ub = np.hstack([self.params[name].ub_fe for name in param_names])

        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self.params[name].constr_matrix_fe,
             self.params[name].constr_lb_fe,
             self.params[name].constr_ub_fe)
            for name in param_names
        ])

    @property
    def beta_names(self) -> List[str]:
        betas = []
        for key, val in self.params:
            if "beta" in key:
                betas.append(key)
        return betas

    @property
    def fevar_size(self) -> int:
        num_fe = 0
        for beta in self.beta_names:
            num_fe += self.params[beta].num_fe
        return num_fe

    @property
    def revar_size(self) -> int:
        return self.params["gamma"].num_fe

    @property
    def ievar_size(self) -> int:
        return self.params["eta"].num_fe

    @property
    def var_sizes(self) -> List[int]:
        return [self.fevar_size, self.revar_size, self.ievar_size]

    @property
    def var_size(self) -> int:
        return sum(self.var_sizes)

    @property
    def femat(self) -> ndarray:
        mats = []
        for beta in self.beta_names:
            mats.append(self.params[beta].design_matrix_fe)
        return np.hstack(mats)

    @property
    def remat(self) -> ndarray:
        return self.params["gamma"].design_matrix_fe

    @property
    def iemat(self) -> ndarray:
        return self.params["eta"].design_matrix_fe

    def get_vars(self, x: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        variables = np.split(x, np.cumsum([self.var_sizes])[:-1])
        beta = variables[0]
        gamma = np.sqrt(variables[1]**2)
        eta = np.sqrt(variables[2]**2)
        return beta, gamma, eta

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, weights: np.ndarray):
        if any(weights < 0. or weights > 1.):
            raise ValueError("Weights are not between 0 and 1.")
        self._w = weights

    # pylint:disable=unbalanced-tuple-unpacking
    def _objective(self, x: ndarray, data: Data) -> ndarray:
        """
        Objective function
        """
        beta, gamma, eta = self.get_vars(x)
        r = data.obs - self.femat.dot(beta)

        v_re = np.sum(self.remat**2*gamma, axis=1)
        v_ie = np.sum(self.iemat**2*eta, axis=1)
        v = data.obs_var + v_re + v_ie
        z = np.sqrt(v_ie)*r/np.sqrt(2.0*v*(data.obs_var + v_re))

        return 0.5 * r ** 2 / v + 0.5 * np.log(v) - log_erfc(z)

    def objective(self, x: ndarray, data: Data) -> float:
        obj = self._objective(x=x, data=data)
        if self.w is not None:
            obj = self.w.dot(obj)
        return np.mean(obj)

    def _gradient(self, x: ndarray, data: Data) -> ndarray:
        beta, gamma, eta = self.get_vars(x)
        r = data.obs - self.femat.dot(beta)

        v_re = np.sum(self.remat ** 2 * gamma, axis=1)
        v_ie = np.sum(self.iemat ** 2 * eta, axis=1)
        v_roe = data.obs_var + v_re
        v = data.obs_var + v_re + v_ie

        z = np.sqrt(v_ie) * r / np.sqrt(2.0 * v * (data.obs_var + v_re))
        x = self.femat

        # Derivative of log erfc
        index = z >= 10.0
        dlerf = np.zeros(z.shape)
        dlerf[index] = -2 * z[index] - 1 / z[index]
        dlerf[~index] = -2 * np.exp(-z[~index]**2) / erfc(z[~index]) / np.sqrt(np.pi)
        grad = np.zeros((beta.size + 2, data.obs.shape[0]))

        grad[:beta.size, ] = x.T * (dlerf*np.sqrt(v_ie/(v_roe*v))/np.sqrt(2) - r/v)
        grad[beta.size, ] = 0.5*(-r**2/v**2 + 1/v + dlerf*r*np.sqrt(v_ie/(v_roe*v))/np.sqrt(2)*(1/v_roe + 1/v))
        grad[-1, ] = 0.5*(-r**2/v**2 + 1/v - dlerf*r*np.sqrt(v_ie/(v_roe*v))/np.sqrt(2)*(1/v_ie - 1/v))

        return grad

    def gradient(self, x: ndarray, data: Data) -> ndarray:
        """
        Computes the gradient.

        :param x:
        :param data:
        :param w: optional weights
        :return:
        """
        grad = self._gradient(x=x, data=data)
        if self.w is not None:
            grad = self.w * grad
        grad = np.sum(grad, axis=1)

        # Take the average because the objective
        # is the mean rather than the sum
        grad = grad / data.obs.shape[0]

        return grad

    # pylint:disable=arguments-differ
    def forward(self, x: ndarray, mat: ndarray = None) -> ndarray:
        mat = self.femat if mat is None else mat
        beta = self.get_vars(x)[0]
        return mat.dot(beta)

    def get_ie(self, x: ndarray, data: Data) -> ndarray:
        """
        Get inefficiency
        """
        beta, gamma, eta = self.get_vars(x)
        r = data.obs - self.femat.dot(beta)

        v_re = np.sum(self.remat**2*gamma, axis=1)
        v_ie = np.sum(self.iemat**2*eta, axis=1)

        return np.maximum(0.0, -eta[0]*r/(data.obs_var + v_re + v_ie))

    def get_re(self, x: ndarray, data: Data) -> ndarray:
        """
        Get random effects
        """
        beta, gamma, _ = self.get_vars(x)
        r = data.obs - self.femat.dot(beta)

        v_re = np.sum(self.remat**2*gamma, axis=1)
        ie = self.get_ie(x, data)

        return gamma[0]*(r + ie)/(data.obs_var + v_re)

    def get_var_init(self, data) -> ndarray:
        """
        Compute the initialization of the variable
        """
        n = len(data.obs)
        beta_init = np.linalg.solve(
            (self.femat.T/data.obs_var).dot(self.femat),
            (self.femat.T/data.obs_var).dot(data.obs)
        )

        # Estimate the residuals
        r = data.obs - self.femat.dot(beta_init)

        # Get the largest residual, this is a crude estimate
        # for the intercept shift required to go through the data
        alpha = np.max(r)
        beta_init += alpha

        # Calculate the first moment
        # E[r_i] = E[u_i] + E[v_i] + E[\epsilon_i] + \alpha
        # This expression is our estimate of \sqrt{2\eta/\pi}
        eta = (np.mean(r) - alpha) ** 2 * np.pi / 2

        # Calculate the second moment
        # (\sum E[r_i^2] - \sum \sigma_i**2)/n =
        #   \gamma + \eta (1 - 2/\pi) + (\alpha + \sqrt{2\eta / \pi})^2
        moment2 = np.sum(r**2 - data.obs_se) / n
        gamma = moment2 - eta * (1 - 2 / np.pi) - (alpha + np.sqrt(2 * eta / np.pi)) ** 2
        # gamma = 1e-5
        # eta = 1e-5
        return np.hstack([beta_init, gamma, eta])
