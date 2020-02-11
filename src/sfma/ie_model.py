# -*- coding: utf-8 -*-
"""
   ie_model
   ~~~~~~~~

   Inefficiency model module.
"""
import numpy as np


class IEModel:
    """Inefficiency Model.
    """
    def __init__(self, model):
        self.model = model

    def init_params(self):
        """Initialize the parameters including
        * v (inefficiency/efficiency)
        * delta (parameter of the distribution of v)
        * ie_pred (prediction from the inefficiency model)
        """
        pass

    def update_params(self):
        """Update related parameters (v, delta, ie_pred).
        """
        pass

    def fit(self):
        """Fit the ie_model parameters.
        """
        pass
