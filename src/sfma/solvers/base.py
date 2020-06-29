import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy

from anml.solvers.interface import CompositeSolver, Solver

from sfma.data import Data


class IterativeSolver(CompositeSolver):

    def __init__(self, solvers_list: List[Solver] = None):
        super().__init__(solvers_list)
        self.reset()

    def reset(self):
        self.data = None
    
    def step(self, data, verbose=True):
        raise NotImplementedError()

    def fit(self, x_init: List[np.ndarray], data: Data, verbose=True, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        self.x_curr = x_init 
        data.y = deepcopy(data.obs)
        data.sigma2 = deepcopy(data.obs_se**2)
        self.errors_hist = []
        itr = 0
        while itr < options['maxiter']:
            if verbose:
                print('-' * 10, f'iter {itr}', '-' * 10)
            self.step(data, verbose)
            self.errors_hist.append(self.error(data))
            if verbose:
                print(f'error = {self.errors_hist[-1]}')
            itr += 1

        self.x_opt = self.x_curr
        self.fun_val_opt = self.error(data)

    def forward(self, x):
        raise NotImplementedError()
    
    def error(self, data, x = None):
        if x is None:
            return np.mean((data.obs - self.predict())**2 / data.obs_se**2)
        else:
            return np.mean((data.obs - self.forward(x))**2 / data.obs_se**2)
