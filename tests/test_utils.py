# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~
    Test `utils` module of `sfma` package.
"""
import numpy as np
import pandas as pd
import pytest
import sfma.utils as utils


@pytest.mark.parametrize('df', [pd.DataFrame({'alpha': np.ones(5),
                                              'beta': np.zeros(5)})])
@pytest.mark.parametrize(('col_names', 'col_shape'),
                         [('alpha', (5,)),
                          ('beta', (5,)),
                          (['alpha'], (5, 1)),
                          (['beta'], (5, 1)),
                          (['alpha', 'beta'], (5, 2)),
                          (None, (5, 0))])
def test_get_columns(df, col_names, col_shape):
    col = utils.get_columns(df, col_names)
    assert col.shape == col_shape


@pytest.mark.parametrize(('col_names', 'ok'),
                         [('col0', True),
                          (['col0', 'col1'], True),
                          ([], True),
                          (None, True),
                          (1, False)])
def test_is_col_names(col_names, ok):
    assert ok == utils.is_col_names(col_names)


@pytest.mark.parametrize('col_names', [None, 'col0', ['col0', 'col1']])
@pytest.mark.parametrize('default', [None, 'col0', ['col0', 'col1']])
def test_input_col_names_default(col_names, default):
    result_col_names = utils.input_col_names(col_names,
                                             default=default)
    if col_names is None:
        assert result_col_names == [] if default is None else default
    else:
        assert result_col_names == col_names


@pytest.mark.parametrize('col_names', [None, 'col0', ['col0', 'col1']])
@pytest.mark.parametrize('full_col_names', [None, ['col2']])
def test_input_col_names_append_to(col_names, full_col_names):
    col_names = utils.input_col_names(col_names,
                                      append_to=full_col_names)
    if full_col_names is not None and col_names:
        assert 'col0' in full_col_names and 'col2' in full_col_names
        if isinstance(col_names, list):
            assert 'col1' in full_col_names
