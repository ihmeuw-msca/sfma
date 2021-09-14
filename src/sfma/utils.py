import numpy as np
from numpy import ndarray
from scipy.special import erfc


def log_erfc(x: ndarray) -> ndarray:
    """Ln erfc function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Ln erfc function values.
    """
    y = np.empty(x.shape, dtype=x.dtype)
    indices = x < 25
    y[indices] = np.log(erfc(x[indices]))
    y[~indices] = -x[~indices]**2 + np.log(1 - 0.5/x[~indices]**2) - \
        np.log(np.sqrt(np.pi)*x[~indices])
    return y


def dlog_erfc(x: np.ndarray) -> np.ndarray:
    """Derivative of ln erfc function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Derivative of ln erfc.
    """
    y = np.zeros(x.shape, dtype=x.dtype)
    indices = x < 25
    y[indices] = -2*np.exp(-x[indices]**2)/(erfc(x[indices])*np.sqrt(np.pi))
    y[~indices] = -2*x[~indices] - 1/x[~indices] + 2/(2*x[~indices]**3 - x[~indices])

    return y


def d2log_erfc(x: np.ndarray) -> np.ndarray:
    """Second order derivative of ln erfc function.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Second order derivative of ln erfc.
    """
    d1 = dlog_erfc(x)
    return -2*x*d1 - d1**2
