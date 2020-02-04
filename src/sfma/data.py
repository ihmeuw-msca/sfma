# -*- coding: utf-8 -*-
"""
    data
    ~~~~
    `data` module of the `sfma` package.
"""
import numpy as np
import pandas as pd
import warnings
from . import utils


class Data:
    """Data structure for easy processing
    """
    def __init__(self, df,
                 obs=None,
                 obs_se=None,
                 covs=None,
                 u_id=None,
                 v_id=None,
                 add_intercept=False):
        """Constructor of Data

        Args:
            df (pandas.DataFrame):
                Dataframe from csv file that store the data.
            obs (str | None, optional):
                Column name that store the observations of the problem.
            obs_se (str | None, optional):
                Column name that store the standard error of the observations.
            covs (list{str} | None, optional):
                List of column names of potential covariates.
            u_id (str | None, optional):
                Column name that store the grouping id of the random effects.
            v_id (str | None, optional):
                Column name that store the grouping id of the
                efficiency/inefficiency.
            add_intercept (bool, optional):
                If `True`, add intercept to the current covariates.
        """
        # pass in data
        self.df = df if u_id is None else df.sort_values(u_id)
        self.obs = utils.get_columns(df, obs)
        self.obs_se = utils.get_columns(df, obs_se)
        self.covs = utils.get_columns(df, covs)
        self.u_id = utils.get_columns(df, u_id)
        self.v_id = utils.get_columns(df, v_id)

        if add_intercept:
            if 'intercept' in self.covs:
                warnings.warn("Intercept already exist, no need to add.")
            else:
                self.covs['intercept'] = np.ones(self.df.shape[0])

        # dimensions
        self.num_obs = self.df.shape[0]
        self.num_covs = self.covs.shape[1]

        # grouping structure
        if self.u_id.empty:
            self.u_id['u_id'] = np.arange(self.num_obs)
        if self.v_id.empty:
            self.v_id['v_id'] = np.arange(self.num_obs)

        self.unique_u_id, \
        self.u_group_sizes = np.unique(self.u_id, return_counts=True)
        self.num_u_groups = self.unique_u_id.size

        self.unique_v_id, \
        self.v_group_sizes = np.unique(self.v_id, return_counts=True)
        self.num_v_groups = self.unique_v_id.size

        # group index
        self.u_group_idx = {
            i: np.where(self.u_id == i)[0]
            for i in self.unique_u_id
        }
        self.v_group_idx = {
            i: np.where(self.v_id == i)[0]
            for i in self.unique_v_id
        }

    def __repr__(self):
        """Summary of the object.
        """
        dimension_summary = [
            "number of observations: %i"%self.num_obs,
            "number of covariates  : %i"%self.num_covs,
            "number of u groups    : %i"%self.num_u_groups,
            "number of v groups    : %i"%self.num_v_groups,
        ]
        return "\n".join(dimension_summary)
