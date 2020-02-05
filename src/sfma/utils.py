# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~
    `utils` module of the `sfma` package.
"""
import numpy as np
import pandas as pd


def get_columns(df, col_names):
    """Return the columns of the given data frame.

    Args:
        df (pandas.DataFrame):
            Given data frame.
        col_names (str | list{str} | None):
            Given column name(s), if is `None`, will return a empty data frame.

    Returns:
        pandas.DataFrame:
            The data frame contains the columns.
    """
    assert isinstance(df, pd.DataFrame)
    if isinstance(col_names, list):
        assert all([isinstance(col_name, str) and col_name in df
                    for col_name in col_names])
    else:
        assert (col_names is None) or (isinstance(col_names, str) and
                                       col_names in df)

    if col_names is None:
        return df[[]].copy()
    else:
        return df[col_names].copy()
