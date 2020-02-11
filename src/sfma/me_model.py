# -*- coding: utf-8 -*-
"""
   me_model
   ~~~~~~~~

   Mixed-Effects model module.
"""
import numpy as np
import pandas as pd
from limetr import LimeTr
from . import utils


class MEModel:
    """Mixed-Effects Model.
    """
    def __init__(self, model):
        """Constructor of the mixed-effects model.

        Args:
            model (sfma.Model):
                Model information.
        """
        self.model = model
        self.model.data.sort_by_u_id()
        self.study_sizes = self.model.data.u_group_sizes.values

        self.x_covs = []
        self.z_covs = []
        self.c_covs = []
        self.x_sizes = []
        self.z_sizes = []
        self.c_sizes = []
        self.x_idx = {}
        self.z_idx = {}
        self.c_idx = {}
        for cov_name, cov_model in self.model.cov_models.items():
            self.x_covs.append(cov_name)
            self.x_sizes.append(cov_model.num_vars)
            self.x_idx.update({
                cov_name: np.arange(sum(self.x_sizes[:-1]),
                                    self.x_sizes[-1])
            })

            if cov_model.add_u:
                self.z_covs.append(cov_name)
                self.z_sizes.append(cov_model.num_vars)
                self.z_idx.update({
                    cov_name: np.arange(sum(self.z_sizes[:-1]),
                                        self.z_sizes[-1])
                })

            if cov_model.num_constraints != 0:
                self.c_covs.append(cov_name)
                self.c_sizes.append(cov_model.num_constraints)
                self.c_idx.update({
                    cov_name: np.arange(sum(self.c_sizes[:-1]),
                                        self.c_sizes[-1])
                })

        self.num_x_vars = sum(self.x_sizes)
        self.num_z_vars = sum(self.z_sizes)
        self.num_vars = self.num_x_vars + self.num_z_vars
        self.num_constraints = sum(self.c_sizes)

        self.x = self.create_design_mat(cov_names=self.x_covs)
        self.z = self.create_design_mat(cov_names=self.z_covs)
        self.c = self.create_constraint_mat(cov_names=self.c_covs)

    def create_design_mat(self, data=None, cov_names=None):
        """Create linear design matrix.

        Keyword Args:
            data (sfma.Data | None, optional):
                If not None, self.data.
            cov_names (list{str}):
                Model names used for design matrix.

        Returns:
            numpy.ndarray:
                Returns the design matrix.
        """
        cov_names = self.x_covs if cov_names is None else cov_names
        data = self.model.data if data is None else data
        design_mat = np.hstack([
            self.model.cov_models[cov_name].create_design_mat(data)
            for cov_name in cov_names
        ])

        return design_mat

    def create_constraint_mat(self, data=None, cov_names=None):
        """Create linear constraint matrix.

        Keyword Args:
            data (sfma.Data | None, optional):
                if not None, self.data.
            cov_names (list{str}):
                Model names used for design matrix.

        Returns:
            numpy.ndarray:
                Returns the constraint matrix.
        """
        cov_names = self.c_covs if cov_names is None else cov_names
        data = self.model.data if data is None else data

        constraint_mat = []
        for cov_name in cov_names:
            if cov_name not in self.c_covs:
                continue
            cov_model = self.model.cov_models[cov_name]
            sub_mat = np.zeros((cov_model.num_constraints, self.num_vars))
            sub_mat[:, self.x_idx[cov_name]] = \
                cov_model.create_constraint_mat(data)
            constraint_mat.append(sub_mat)

        return np.vstack(constraint_mat)

    def fit(self):
        """Fit mixed effects model.
        """
        self.model.data.sort_by_u_id()
        # dimensions for limetr
        n = self.study_sizes
        k_beta = self.num_x_vars
        k_gamma = self.num_z_vars

        # input data
        y = self.model.data.obs - self.model.data.ie_pred
        s = self.model.data.obs_se

        # design matrices
        def fun(var):
            return self.x.dot(var)

        def jfun(var):
            return self.x

        uprior = np.array([[-np.inf]*self.num_x_vars + [0.0]*self.num_z_vars,
                           [np.inf]*self.num_vars])
        if self.z.size == 0:
            z = np.ones(self.model.data.num_obs)
            uprior[:, -1] = 0.0
        else:
            z = self.z

        if self.c.size == 0:
            cfun, jcfun, cvec = None, None, None
        else:
            def cfun(var):
                return self.c.dot(var)

            def jcfun(var):
                return self.c

            cvec = np.array([[-np.inf]*self.num_constraints,
                             [0.0]*self.num_constraints])

        lt = LimeTr(n, k_beta, k_gamma, y, fun, jfun, z, S=s,
                    uprior=uprior,
                    C=cfun,
                    JC=jcfun,
                    c=cvec,
                    inlier_percentage=self.model.inlier_pct)

        beta, gamma, w = lt.fitModel()
        u = lt.estimateRE()
        me_pred = y - self.x.dot(beta) - np.sum(z*gamma, axis=1)

    def init_params(self):
        """Initialize the parameters.
        """
        self.update_params(beta=np.zeros(self.num_x_vars),
                           gamma=np.zeros(self.num_z_vars),
                           u=np.zeros((self.study_sizes.size,
                                       self.num_z_vars)),
                           weights=np.ones(self.x.shape[0]),
                           me_pred=np.zeros(self.x.shape[0]))

    def update_params(self,
                      beta=None,
                      gamma=None,
                      u=None,
                      weights=None,
                      me_pred=None):
        """Update the parameter in the model object.

        Keyword Args:
            beta (numpy.ndarray | None, optional):
                Covariates multiplier for the x covs.
            gamma (numpy.ndarray | None, optional):
                Variance of the covariates multiplier for the random z covs.
            u (numpy.ndarray | None, optional):
                Random effect for each study.
            weights (numpy.ndarray | None, optional):
                Trimming weights.
            me_pred (numpy.ndarray | None, optional):
                Prediction for the mixed-effects model.
        """
        if beta is not None:
            self.model.beta = {
                cov_name: beta[self.x_idx[cov_name]]
                for cov_name in self.x_covs
            }

        if gamma is not None:
            self.model.gamma = {
                cov_name: gamma[self.z_idx[cov_name]]
                for cov_name in self.z_covs
            }

        if u is not None:
            self.u = pd.DataFrame(u, index=self.model.data.u_group_sizes.index)
            # TODO


