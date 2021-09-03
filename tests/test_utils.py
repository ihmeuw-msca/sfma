"""
Test utils module
"""
import pytest
import numpy as np
from scipy.special import erfc
from sfma.utils import log_erfc, dlog_erfc


def test_log_erfc_normal_value():
    x = np.zeros(1)
    assert np.allclose(log_erfc(x), 0.0)


def test_log_erfc_extreme_value():
    x0 = np.array([25.0 - 1e-10])
    x1 = np.array([25.0])
    assert np.allclose(log_erfc(x0), log_erfc(x1))


def test_dlog_erfc():
    x = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0])
    my_value = dlog_erfc(x)
    tr_value = np.array([
        log_erfc(np.array([x[i] + 1e-10j]))[0].imag/1e-10
        for i in range(x.size)
    ])
    assert np.allclose(my_value, tr_value, rtol=1e-3)
