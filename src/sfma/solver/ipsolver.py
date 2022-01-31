"""
Customized Interior Point Solver
================================

Solver class solves large scale sparse least square problem with linear
constraints.
"""
from typing import Callable, List

import numpy as np
from scipy.optimize import LinearConstraint


class IPSolver:
    """Interior point solver for large sparse quadratic system with linear
    constraints.

    Parameters
    ----------
    grad : Callable
        Gradient function of the optimization problem.
    hess : Callable
        Hessian function of the optimization problem.
    linear_constraints : Optional[LinearConstraint], optional
        Linear constraints for the problem. Default to be `None`. If it is
        `None`, solver will use a simple linear solve.

    Attributes
    ----------
    grad : Callable
        Gradient function of the optimization problem.
    hess : Callable
        Hessian function of the optimization problem.
    linear_constraints : Optional[LinearConstraint]
        Linear constraints for the problem.
    c_mat : Optional[np.ndarray]
        Constraint matrix, when `linear_constraints` is `None`, `c_mat` will be
        `None` as well.
    c_vec : Optional[np.ndarray]
        Constraint vector, when `linear_constraints` is `None`, `c_mat` will be
        `None` as well.

    Methods
    -------
    get_kkt(p, mu)
        Get the KKT system.
    update_params(p, dp, mu, scale=0.99)
        Update parameters with line search.
    minimize(xtol=1e-8, gtol=1e-8, max_iter=100, mu=1.0, scale_mu=0.1,
             scale_step=0.99, verbose=False)
        Minimize optimization objective over constraints.
    """

    def __init__(self,
                 grad: Callable,
                 hess: Callable,
                 linear_constraints: LinearConstraint):
        self.grad = grad
        self.hess = hess
        self.linear_constraints = linear_constraints

        mat = self.linear_constraints.A
        index = ~np.isclose(mat, 0.0).all(axis=1)
        mat = mat[index]
        scale = np.abs(mat).max(axis=1)
        mat = mat / scale[:, np.newaxis]
        lb = self.linear_constraints.lb[index] / scale
        ub = self.linear_constraints.ub[index] / scale

        self.c_mat = np.vstack([-mat[~np.isneginf(lb)], mat[~np.isposinf(ub)]])
        self.c_vec = np.hstack([-lb[~np.isneginf(lb)], ub[~np.isposinf(ub)]])

    def get_kkt(self,
                p: List[np.ndarray],
                mu: float) -> List[np.ndarray]:
        """Get the KKT system.

        Parameters
        ----------
        p : List[np.ndarray]
            A list a parameters, including x, s, and v, where s is the slackness
            variable and v is the dual variable for the constraints.
        mu : float
            Interior point method barrier variable.

        Returns
        -------
        List[np.ndarray]
            The KKT system with three components.
        """
        return [
            self.c_mat.dot(p[0]) + p[1] - self.c_vec,
            p[1]*p[2] - mu,
            self.grad(p[0]) + self.c_mat.T.dot(p[2])
        ]

    def update_params(self,
                      p: List[np.ndarray],
                      dp: List[np.ndarray],
                      mu: float) -> float:
        """Update parameters with line search.

        Parameters
        ----------
        p : List[np.ndarray]
            A list a parameters, including x, s, and v, where s is the slackness
            variable and v is the dual variable for the constraints.
        dp : List[np.ndarray]
            A list of direction for the parameters.
        mu : float
            Interior point method barrier variable.

        Returns
        -------
        float
            The step size in the given direction.
        """
        c = 0.01
        a = 1.0
        for i in [1, 2]:
            indices = dp[i] < 0.0
            if not any(indices):
                continue
            a = 0.99*np.minimum(a, np.min(-p[i][indices] / dp[i][indices]))

        f_curr = self.get_kkt(p, mu)
        gnorm_curr = np.max(np.abs(np.hstack(f_curr)))

        for i in range(20):
            p_next = [v.copy() for v in p]
            for i in range(len(p)):
                p_next[i] += a * dp[i]
            f_next = self.get_kkt(p_next, mu)
            gnorm_next = np.max(np.abs(np.hstack(f_next)))
            if gnorm_next <= (1 - c*a)*gnorm_curr:
                break
            a *= 0.9
        return a, p_next

    def minimize(self,
                 x0: np.ndarray,
                 xtol: float = 1e-8,
                 gtol: float = 1e-8,
                 mtol: float = 1e-6,
                 max_iter: int = 100,
                 mu: float = 1.0,
                 update_mu_every: int = 5,
                 scale_mu: float = 0.5,
                 verbose: bool = False) -> np.ndarray:
        """Minimize optimization objective over constraints.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for the solution.
        xtol : float, optional
            Tolerance for the differences in `x`, by default 1e-8.
        gtol : float, optional
            Tolerance for the KKT system, by default 1e-8.
        max_iter : int, optional
            Maximum number of iterations, by default 100.
        mu : float, optional
            Initial interior point bairrier parameter, by default 1.0.
        scale_mu : float, optional
            Shrinkage factor for mu updates, by default 0.1
        verbose : bool, optional
            Indicator of if print out convergence history, by default False

        Returns
        -------
        np.ndarray
            Solution vector.
        """

        # initialize the parameters
        p = [
            x0,
            np.ones(self.c_vec.size),
            np.ones(self.c_vec.size)
        ]

        f = self.get_kkt(p, mu)
        gnorm = np.max(np.abs(np.hstack(f)))
        xdiff = 1.0
        step = 1.0
        counter = 0

        if verbose:
            print(f"{type(self).__name__}:")
            print(f"{counter=:3d}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}, "
                  f"{mu=:.2e}")

        while ((gnorm > gtol and xdiff > xtol and counter < max_iter) or
               (mu > mtol)):
            counter += 1

            # cache convenient variables
            sv_vec = p[2] / p[1]
            sf2_vec = f[1] / p[1]
            csv_mat = self.c_mat*sv_vec[:, None]

            # compute all directions
            mat = self.hess(p[0]) + csv_mat.T.dot(self.c_mat)
            inv_mat = np.linalg.pinv(mat)
            vec = -f[2] + self.c_mat.T.dot(sf2_vec - sv_vec*f[0])
            dx = inv_mat.dot(vec)
            ds = -f[0] - self.c_mat.dot(dx)
            dv = -sf2_vec - sv_vec*ds
            dp = [dx, ds, dv]

            # get step size
            step, p = self.update_params(p, dp, mu)

            # update mu
            if counter % update_mu_every == 0:
                mu = max(scale_mu*mu, 0.1*p[1].dot(p[2])/len(p[1]))

            # update f and gnorm
            f = self.get_kkt(p, mu)
            gnorm = np.max(np.abs(np.hstack(f)))
            xdiff = step*np.max(np.abs(dp[0]))

            if verbose:
                print(f"{counter=:3d}, {gnorm=:.2e}, {xdiff=:.2e}, "
                      f"{step=:.2e}, {mu=:.2e}")

        return p[0]
