# -*- coding: utf-8 -*-
"""
    data
    ~~~~
    `data` module of the `sfma` package.
"""
import numpy as np
from . import utils


class Data:
    """Data structure for easy processing
    """
    def __init__(self, df,
                 col_obs=None,
                 col_obs_se=None,
                 col_covs=None,
                 col_u_id=None,
                 col_v_id=None,
                 add_intercept=False):
        """Constructor of Data

        Args:
            df (pandas.DataFrame):
                Dataframe from csv file that store the data.
            col_obs (str | None, optional):
                Column name that store the observations of the problem.
            col_obs_se (str | None, optional):
                Column name that store the standard error of the observations.
            col_covs (list{str} | None, optional):
                List of column names of potential covariates.
            col_u_id (str | None, optional):
                Column name that store the grouping id of the random effects.
            col_v_id (str | None, optional):
                Column name that store the grouping id of the
                efficiency/inefficiency.
            add_intercept (bool, optional):
                If `True`, add intercept to the current covariates.
        """
        # pass in column names
        self.col_names = []
        self.col_obs = utils.input_col_names(col_obs,
                                             append_to=self.col_names)
        self.col_obs_se = utils.input_col_names(col_obs_se,
                                                append_to=self.col_names)
        self.col_covs = utils.input_col_names(col_covs,
                                              append_to=self.col_names)
        self.col_u_id = utils.input_col_names(col_u_id,
                                              append_to=self.col_names,
                                              default='u_id')
        self.col_v_id = utils.input_col_names(col_v_id,
                                              append_to=self.col_names,
                                              default='v_id')

        if add_intercept and 'intercept' not in self.col_names:
            self.col_covs.append('intercept')
            self.col_names.append('intercept')

        # pass in data frame
        self.df = None
        self.update_df(df)

    def update_df(self, df):
        """Update the current data frame.

        Args:
            df (pandas.DataFrame):
                New input data frame.
        """
        # add columns if necessary
        df = df.copy()
        if 'intercept' in self.col_covs and 'intercept' not in df:
            df['intercept'] = np.ones(df.shape[0])
        if self.col_u_id not in df:
            df[self.col_u_id] = np.arange(df.shape[0])
        if self.col_v_id not in df:
            df[self.col_v_id] = np.arange(df.shape[0])

        self.df = df[self.col_names].copy()

    @property
    def num_obs(self):
        return self.df.shape[0]

    @property
    def num_covs(self):
        return self.covs.shape[1]

    @property
    def num_u_groups(self):
        return self.u_group_sizes.size

    @property
    def num_v_groups(self):
        return self.v_group_sizes.size

    @property
    def obs(self):
        return utils.get_columns(self.df, self.col_obs)

    @property
    def obs_se(self):
        return utils.get_columns(self.df, self.col_obs_se)

    @property
    def covs(self):
        return utils.get_columns(self.df, self.col_covs)

    @property
    def u_id(self):
        return utils.get_columns(self.df, self.col_u_id)

    @property
    def v_id(self):
        return utils.get_columns(self.df, self.col_v_id)

    @property
    def u_group_sizes(self):
        return self.u_id.value_counts()

    @property
    def v_group_sizes(self):
        return self.v_id.value_counts()

    @property
    def u_group_idx(self):
        return {
            i: np.where(self.u_id == i)[0]
            for i in self.u_group_sizes.index
        }

    @property
    def v_group_idx(self):
        return {
            i: np.where(self.v_id == i)[0]
            for i in self.u_group_sizes.index
        }

    def sort_by_u_id(self):
        self.df.sort_values(self.col_u_id, inplace=True)

    def sort_by_v_id(self):
        self.df.sort_values(self.col_v_id, inplace=True)

    def __repr__(self):
        """Summary of the object.
        """
        dimension_summary = [
            "number of observations: %i" % self.num_obs,
            "number of covariates  : %i" % self.num_covs,
            "number of u groups    : %i" % self.num_u_groups,
            "number of v groups    : %i" % self.num_v_groups,
        ]
        return "\n".join(dimension_summary)
