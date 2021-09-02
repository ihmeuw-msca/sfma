# pylint:disable=no-name-in-module
from typing import List, Tuple
import numpy as np
from numpy import ndarray
from scipy.special import erfc

from anml.parameter.utils import combine_constraints


def build_linear_constraint(constraints: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    mats, lbs, ubs = zip(*constraints)
    C, c_lb, c_ub = combine_constraints(mats, lbs, ubs)
    if np.count_nonzero(C) == 0:
        C, c_lb, c_ub = None, None, None
    return C, c_lb, c_ub


def log_erfc(x: ndarray) -> ndarray:
    """
    Ln Erfc function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Ln Erfc function values.
    """
    y = np.empty(x.shape, dtype=x.dtype)
    indices0 = x < 25
    indices1 = ~indices0
    y[indices0] = np.log(erfc(x[indices0]))
    y[indices1] = -x[indices1]**2 + np.log(1.0 - 0.5/x[indices1]**2) - \
        np.log(np.sqrt(np.pi)*x[indices1])
    return y
