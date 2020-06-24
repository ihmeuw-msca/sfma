# -*- coding: utf-8 -*-
"""
    data
    ~~~~
    `data` module of the `sfma` package.
"""
import numpy as np
import pandas as pd
from typing import List, Union
from dataclasses import dataclass
from copy import deepcopy

from anml.parameter.parameter import ParameterSet
from anml.parameter.variables import Variable


@dataclass
class DataSpecs:

    col_obs: str 
    col_obs_se: str


class Data:

    def __init__(self, data_specs: DataSpecs, params: List[ParameterSet] = None):
        self.data_specs = data_specs
        self.params = params
    
    def process(self, df: pd.DataFrame):
        self.obs = df[self.data_specs.col_obs].to_numpy()
        self.obs_se = df[self.data_specs.col_obs_se].to_numpy()
        self.y = df[self.data_specs.col_obs].to_numpy()
        self.sigma2 = df[self.data_specs.col_obs_se].to_numpy()**2

    def reset(self):
        self.sigma2 = (self.obs_se)**2
        self.y = deepcopy(self.obs)


