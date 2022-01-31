from typing import Callable, List, Optional

from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize


class SPSolver:
    """Scipy solver.

    """

    def __init__(self,
                 objective: Callable,
                 gradient: Callable,
                 hessian: Callable,
                 linear_constraints: Optional[List[LinearConstraint]] = None):
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        if linear_constraints is None:
            self.linear_constraints = []
        else:
            self.linear_constraints = linear_constraints

    def minimize(self, x0: NDArray, **options) -> NDArray:
        result = minimize(self.objective, x0,
                          method="trust-constr",
                          jac=self.gradient,
                          hess=self.hessian,
                          constraints=self.linear_constraints,
                          options=options)
        return result.x
