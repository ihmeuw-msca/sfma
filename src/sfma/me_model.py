# -*- coding: utf-8 -*-
"""
   me_model
   ~~~~~~~~

   Mixed-Effects model module.
"""
import numpy as np
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
        self.y = self.model.data.obs.values
        self.s = self.model.data.obs_se.values

        self.fixed_models = list(self.model.cov_models.keys())
        self.random_models = [
            cov_model
            for cov_model in self.model.cov_models
            if self.model.cov_models[cov_model].add_u
        ]
        fixed_var_sizes = [
            self.model.cov_models[model_name].num_fixed_effects
            for model_name in self.fixed_models
        ]
        self.num_fixed_vars = sum(fixed_var_sizes)
        fixed_var_idx = utils.sizes_to_indices(fixed_var_sizes)
        self.fixed_var_idx = {
            self.fixed_models[i]: fixed_var_idx[i]
            for i in range(len(self.fixed_models))
        }
        random_var_sizes = [
            self.model.cov_models[model_name].num_random_effects
            for model_name in self.random_models
        ]
        self.num_random_vars = sum(random_var_sizes)
        random_var_idx = utils.sizes_to_indices(random_var_sizes)
        self.random_var_idx = {
            self.random_models[i]: random_var_idx[i]
            for i in range(len(self.random_models))
        }

        self.x = self.create_design_mat(model_names=self.fixed_models)
        self.z = self.create_design_mat(model_names=self.random_models)
        self.c = self.create_constraint_mat()


    def create_design_mat(self, data=None, model_names=None):
        """Create linear design matrix.

        Keyword Args:
            data (sfma.Data | None, optional):
                If not None, self.data.
            model_names (list{str}):
                Model names used for design matrix.

        Returns:
            numpy.ndarray:
                Returns the design matrix.
        """
        model_names = [] if model_names is None else model_names
        data = self.model.data if data is None else data
        design_mat = np.hstack([
            self.model.cov_models[model_name].create_design_mat(data)
            for model_name in model_names
        ])

        return design_mat

    def create_constraint_mat(self, data=None):
        """Create linear constraint matrix.

        Keyword Args:
            data (sfma.Data | None, optional):
                if not None, self.data.

        Returns:
            numpy.ndarray:
                Returns the constraint matrix.
        """
        data = self.model.data if data is None else data

        constraint_mat = []
        for cov_name, cov_model in self.model.cov_models.items():
            if cov_model.num_constraints == 0:
                continue
            sub_mat = np.zeros((cov_model.num_constraints,
                                self.num_fixed_vars))
            sub_mat[:, self.fixed_var_idx[cov_name]] = \
                cov_model.create_constraint_mat(data)
            constraint_mat.append(sub_mat)

        return np.vstack(constraint_mat)

    def fun(self, var):
        return self.model.x.dot(var)

    def jfun(self, var):
        return self.model.x.values
