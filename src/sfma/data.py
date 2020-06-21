# -*- coding: utf-8 -*-
"""
    data
    ~~~~
    `data` module of the `sfma` package.
"""
import numpy as np
import pandas as pd
from typing import List, Union

from anml.data.data_specs import DataSpecs
from anml.parameter.parameter import ParameterSet
from anml.parameter.variables import Variable


class Data:

    def __init__(self, data_specs: DataSpecs, params: List[ParameterSet]):
        self.data_specs = data_specs
        self.params = params
    
    def process(self, df: pd.DataFrame):
        self.data_specs._validate_df(df)
        self.obs = df[self.data_specs.col_obs].to_numpy()
        self.obs_se = df[self.data_specs.col_obs_se].to_numpy()
        self.y = df[self.data_specs.col_obs].to_numpy()

