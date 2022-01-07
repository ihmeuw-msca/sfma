from typing import Callable, Optional, Protocol

import numpy as np
from numpy.typing import NDArray


class Projector(Protocol):
    def project(self,
                p: NDArray,
                x0: Optional[NDArray] = None,
                **options) -> None:
        """Project point p onto a geometric set."""


class PGSolver:
    """Projected gradient descent solver.

    """

    def __init__(self,
                 objective: Callable,
                 gradient: Callable,
                 hessian: Callable,
                 projector: Projector):
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.projector = projector

    def minimize(self,
                 x0: NDArray,
                 xtol: float = 1e-8,
                 max_iter: int = 100,
                 verbose: bool = False) -> NDArray:
        """Minimize optimization objective over constraints.

        """
        x = self.projector.project(x0)
        obj = self.objective(x0)
        xdiff = 1.0
        counter = 0

        if verbose:
            print(f"{type(self).__name__}:")
            print(f"{counter=:3d}, {obj=:.2e}, {xdiff=:.2e}")

        while (xdiff > xtol) and (counter < max_iter):
            counter += 1

            g = self.gradient(x)
            h = self.hessian(x)
            step = 1.0/np.linalg.norm(h, ord=2)

            x_new = self.projector.project(x - step*g, x0=x)
            xdiff = np.linalg.norm(x_new - x, ord=2) / step

            x = x_new
            obj = self.objective(x)

            if verbose:
                print(f"{counter=:3d}, {obj=:.2e}, {xdiff=:.2e}")

        return x
