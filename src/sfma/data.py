# -*- coding: utf-8 -*-
"""
    data
    ~~~~
    `data` module of the `sfma` package.
"""
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataSpecs:

    col_obs: str
    col_obs_se: str


class Data:

    def __init__(self, data_specs: DataSpecs):
        self.data_specs = data_specs
        self.obs = None
        self.obs_se = None
        self.obs_var = None

    def process(self, df: pd.DataFrame):
        self.obs = df[self.data_specs.col_obs].to_numpy()
        self.obs_se = df[self.data_specs.col_obs_se].to_numpy()
        self.obs_var = self.obs_se**2

    @property
    def num_obs(self) -> int:
        return 0 if self.obs is None else self.obs.size
