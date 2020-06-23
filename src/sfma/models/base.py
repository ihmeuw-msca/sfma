
import numpy as np
import pandas as pd

from anml.models.interface import Model
from anml.parameter.parameter import ParameterSet

from sfma.data import Data


class LinearModel(Model):

    def __init__(self, param_set_processed: ParameterSet = None):
        super().__init__()
        if param_set_processed is not None:
            self.param_set = param_set_processed
        else:
            self._param_set = None

    def forward(self, x, mat=None):
        if mat is None:
            return np.dot(self.design_matrix, x)
        else:
            return np.dot(mat, x)

    def _prerun_check(self, x):
        if len(x) != self.x_dim:
            raise TypeError(f'length of x = {len(x)} is not equal to the number of unknowns = {self.x_dim}.')

    def init_model(self):
        raise NotImplementedError()

    @property
    def param_set(self):
        return self._param_set

    @param_set.setter
    def param_set(self, param_set_processed: pd.DataFrame):
        self._param_set = param_set_processed
        self.init_model()

