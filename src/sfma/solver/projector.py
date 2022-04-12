from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, brentq
from sfma.solver.ipsolver import IPSolver


def proj_csimplex(w: np.ndarray, h: int) -> np.ndarray:
    """Project onto a capped simplex.

    Parameters
    ----------
    w : np.ndarray
        Vector to be projected.
    h : int
        Target sum of projected vector.

    Returns
    -------
    np.ndarray
        Projected vector.
    """
    a, b = np.min(w) - 1.0, np.max(w) - 0.0

    def f(x):
        return np.sum(np.maximum(np.minimum(w - x, 1.0), 0.0)) - h

    x = brentq(f, a, b)
    return np.maximum(np.minimum(w - x, 1.0), 0.0)


class PolyProjector:
    """Projection onto a polyhedron.

    """

    def __init__(self, linear_constraint: LinearConstraint):
        self.linear_constraint = linear_constraint
        self.p = None

    def objective(self, x: NDArray) -> float:
        return 0.5*np.sum((x - self.p)**2)

    def gradient(self, x: NDArray) -> NDArray:
        return x - self.p

    def hessian(self, x: NDArray) -> NDArray:
        return np.identity(x.size)

    def project(self,
                p: NDArray,
                x0: Optional[NDArray] = None,
                **options) -> NDArray:
        self.p = p
        x0 = np.ones(p.size) if x0 is None else x0
        solver = IPSolver(self.gradient, self.hessian, self.linear_constraint)
        return solver.minimize(x0, **options)
