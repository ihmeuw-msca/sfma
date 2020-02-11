# -*- coding: utf-8 -*-
"""
    model
    ~~~~~
    `model` module of the `sfma` package.
"""
import numpy as np
import pandas as pd
from . import utils
from .me_model import MEModel
from .ie_model import IEModel


class Model:
    """Model class for sfma package.
    """
    def __init__(self, data, cov_models,
                 v_distribution='half_normal',
                 v_type='upper',
                 inlier_pct=1.0):
        """Constructor of the Model object.

        Args:
            data (sfma.Data):
                Data structure regarding to the problem.
            cov_model (list{sfma.CovModel}):
                A list of covariates models.

        Keyword Args:
            v_distribution (str, optional):
                Distribution of the inefficiency/efficiency.
            v_type (str, optional):
                If 'upper' fitting the upper frontier and if 'lower' fitting the
                lower frontier.
            inlier_pct (float, optional):
                Float number between 0 and 1 indicates the percentage of the
                inliers.
        """
        # pass in data
        self.data = data
        self.cov_models = {
            cov_model.cov_name: cov_model
            for cov_model in cov_models
        }
        self.v_distribution = v_distribution
        self.v_type = v_type
        self.inlier_pct = inlier_pct

        # place holder for the solutions
        self.beta = None
        self.gamma = None
        self.delta = None
        self.u = None
        self.v = None

        # create mixed effects models and inefficiency model
        self.me_model = MEModel(self)
        self.ie_model = IEModel(self)

        # initialize the parameter
        self.me_model.init_params()
        self.ie_model.init_params()


    def fit(self):
        """Fit the model.
        """
        pass
