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
