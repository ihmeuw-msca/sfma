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
        pandas.DataFrame | pandas.Series:
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
        return df[[]]
    else:
        return df[col_names]


def is_col_names(col_names):
    """Check variable type fall into the column name category.

    Args:
        col_names (str | list{str} | None):
            Column names candidate.

    Returns:
        bool:
            if `col_name` is either str, list{str} or None
    """
    ok = isinstance(col_names, (str, list)) or col_names is None
    if isinstance(col_names, list):
        ok = ok and all([isinstance(col_name, str)
                         for col_name in col_names])
    return ok


def input_col_names(col_names,
                    append_to=None,
                    default=None):
    """Process the input column name.

    Args:
        col_names (str | list{str} | None):
            The input column name(s).
        append_to (list{str} | None, optional):
            A list keep track of all the column names.
        default (str | list{str} | None, optional):
            Default value when `col_name` is `None`.

    Returns:
        str | list{str}:
            The name of the column(s)
    """
    assert is_col_names(col_names)
    assert is_col_names(append_to)
    assert is_col_names(default)
    default = [] if default is None else default
    col_names = default if col_names is None else col_names

    if isinstance(col_names, list):
        col_names = col_names.copy()

    if col_names is not None and append_to is not None:
        if isinstance(col_names, str):
            append_to.append(col_names)
        else:
            append_to += col_names
    return col_names


def sizes_to_indices(sizes):
    """Converting sizes to corresponding indices.
    Args:
        sizes (numpy.dnarray):
            An array consist of non-negative number.
    Returns:
        list{range}:
            List the indices.
    """
    indices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        indices.append(np.arange(a, b))
        a += size

    return indices
