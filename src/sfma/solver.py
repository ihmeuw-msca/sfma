import numpy as np

from anml.solvers.interface import Solver, CompositeSolver
from anml.solvers.base import ScipyOpt


class SimpleSolver(CompositeSolver):

    def fit(self, x_init, data, options=dict(solver_options=dict())):
        self.solvers[0].fit(x_init, data, options)
        if len(self.solvers) >= 2:
            # compute vs
            data.y = self.solvers[0].predict() - data.obs
            self.solvers[1].model.gammas = [self.solvers[0].x_opt[-1]]
            self.solvers[1].fit(data=data)
        if len(self.solvers) == 3:
            # computes us
            self.solvers[2].model.gammas = [self.solvers[0].x_opt[-2]]
            data.y = data.obs - self.solvers[0].predict() + self.solvers[1].x_opt
            self.solvers[2].fit(data=data)
        self.x_opt = [solver.x_opt for solver in self.solvers]

    def predict(self):
        return self.solvers[0].predict()
        
