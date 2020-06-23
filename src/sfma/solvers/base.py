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

    def _set_params(self):
        assert len(self.data.params) == len(self.solvers)
        for param_set, solver in zip(self.data.params, self.solvers):
            solver.model.param_set = param_set
    
    def step(self, data, verbose=True):
        raise NotImplementedError()

    @property
    def x_curr(self):
        raise NotImplementedError()

    @x_curr.setter
    def x_curr(self, x):
        raise NotImplementedError()

    def fit(self, x_init: List[np.ndarray], data: Data, set_params: bool = True, verbose=True, options: Optional[Dict[str, str]] = dict(maxiter=100, tol=1e-5)):
        self.x_curr = x_init 
        
        data.y = deepcopy(data.obs)
        
        if set_params:
            self._set_params()
        
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
            x = self.x_curr
        return np.mean((data.obs - self.forward(x))**2 / data.obs_se**2)
