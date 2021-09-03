import numpy as np
from numpy import ndarray
from scipy.special import erfc


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


def dlog_erfc(x: np.ndarray) -> np.ndarray:
    index = x >= 10.0
    y = np.zeros(x.shape)
    y[index] = -2 * x[index] - 1 / x[index]
    y[~index] = -2 * np.exp(-x[~index]**2) / erfc(x[~index]) / np.sqrt(np.pi)
    return y
