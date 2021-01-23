from typing import Tuple, List
import numpy as np
from numpy import ndarray

from anml.models.interface import Model
from anml.parameter.parameter import Parameter

from sfma.data import Data
from sfma.models.utils import build_linear_constraint, log_erfc


class MarginalModel(Model):
    """Marginal model for stochastic frontier.
    """

    def __init__(self, params: List[Parameter]):
        super().__init__()
        if not all([isinstance(param, Parameter) for param in params]):
            raise TypeError("params must be a list of Parameter.")
        param_names = [param.param_name for param in params]
        self.param_names = ["beta", "gamma", "eta"]
        if not all([name in self.param_names for name in param_names]):
            raise ValueError("MarginalModel requires parameter beta, gamma and eta.")
        self.params = {
            param.param_name: param
            for param in params
        }

        # extract constraints information
        self.lb = np.hstack([self.params[name].lb_fe for name in self.param_names])
        self.ub = np.hstack([self.params[name].ub_fe for name in self.param_names])

        self.C, self.c_lb, self.c_ub = build_linear_constraint([
            (self.params[name].constr_matrix_fe,
             self.params[name].constr_lb_fe,
             self.params[name].constr_ub_fe)
            for name in self.param_names
        ])

    @property
    def fevar_size(self) -> int:
        return self.params["beta"].num_fe

    @property
    def revar_size(self) -> int:
        return self.params["gamma"].num_fe

    @property
    def ievar_size(self) -> int:
        return self.params["eta"].num_fe

    @property
    def var_sizes(self) -> int:
        return [self.fevar_size, self.revar_size, self.ievar_size]

    @property
    def var_size(self) -> int:
        return sum(self.var_sizes)

    @property
    def femat(self) -> ndarray:
        return self.params["beta"].design_matrix_fe

    @property
    def remat(self) -> ndarray:
        return self.params["gamma"].design_matrix_fe

    @property
    def iemat(self) -> ndarray:
        return self.params["eta"].design_matrix_fe

    def get_vars(self, x: ndarray) -> Tuple[ndarray]:
        variables = np.split(x, np.cumsum([self.var_sizes])[:-1])
        beta = variables[0]
        gamma = variables[1]
        eta = np.sqrt(variables[2]**2)
        return beta, gamma, eta

    # pylint:disable=unbalanced-tuple-unpacking
    def objective(self, x: ndarray, data: Data) -> float:
        """
        Objective function
        """
        beta, gamma, eta = self.get_vars(x)
        r = data.obs - self.femat.dot(beta)

        v_re = np.sum(self.remat**2*gamma, axis=1)
        v_ie = np.sum(self.iemat**2*eta, axis=1)
        v = data.obs_var + v_re + v_ie
        z = np.sqrt(v_ie)*r/np.sqrt(2.0*v*(data.obs_var + v_re))

        return np.mean(0.5*r**2/v + 0.5*np.log(v) - log_erfc(z))

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
        beta_init = np.linalg.solve(
            (self.femat.T/data.obs_var).dot(self.femat),
            (self.femat.T/data.obs_var).dot(data.obs)
        )
        gamma_init = np.zeros(self.revar_size)
        eta_init = np.zeros(self.ievar_size)
        return np.hstack([beta_init, gamma_init, eta_init])
