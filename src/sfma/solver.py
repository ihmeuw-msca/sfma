import numpy as np

from anml.solvers.interface import Solver, CompositeSolver
from anml.solvers.base import IPOPTSolver


class SimpleSolver(CompositeSolver):

    def fit(self, x_init, data, options=dict(solver_options=dict(max_iter=200))):
        self.solvers[0].fit(x_init, data, options)
        if len(self.solvers) >= 2:
            # compute vs
            data.y = self.solvers[0].predict() - data.obs
            self.solvers[1].fit(data=data)
        if len(self.solvers) == 3:
            # computes us
            data.y = data.obs - self.solvers[0].predict() + self.solvers[1].x_opt
            self.solvers[2].fit(data=data)
        self.x_opt = [solver.x_opt for solver in self.solvers]

    def predict(self):
        return self.solvers[0].predict()
        
