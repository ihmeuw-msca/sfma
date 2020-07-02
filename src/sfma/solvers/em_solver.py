import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
import scipy

from anml.solvers.interface import Solver
from anml.solvers.base import IPOPTSolver, ClosedFormSolver, ScipyOpt

from sfma.data import Data
from sfma.models.marginal import LinearMarginal
from sfma.models.maximal import UModel, VModel
from sfma.solvers.base import IterativeSolver
from sfma.solvers.block_solver import BlockSolver


class EMSolver(BlockSolver):

    def __init__(self, solvers_list: List[Solver]):
        if len(solvers_list) != 2:
            raise RuntimeError(f'This EM solver only works for cases with beta solver and v solver.')
        super().__init__([solvers_list[0], None, solvers_list[1]])

    def v_step(self, data):
        data.y = self.beta_solver.predict() - data.obs
        self.v_solver.model.gammas = self.x_curr[-1]
        self.v_solver.fit(self.x_curr[self.blocks_divider[1]], data)
        mu_v = self.v_solver.x_opt
        sigma2_v = 1 / (np.diag(np.dot(self.v_solver.model.Z.T / data.sigma2, self.v_solver.model.Z)) + 1.0/self.v_solver.model.gammas_padded)
        alpha = -mu_v / np.sqrt(sigma2_v)
        phi_alpha = 1/np.sqrt(2*np.pi) * np.exp(- alpha**2 / 2)
        Phi_alpha = (1 + scipy.special.erf(alpha / np.sqrt(2))) / 2
        z = 1 - Phi_alpha
        vs_curr = mu_v + phi_alpha * np.sqrt(sigma2_v) / z
        self.x_curr[self.blocks_divider[1]] = vs_curr
        sigma2_v_expect = (1 + alpha * phi_alpha / z - (phi_alpha / z)**2) * sigma2_v
        self.x_curr[-1] = [np.mean(vs_curr**2 + sigma2_v_expect)]

        data.sigma2 = (
            np.sum(data.y**2) + 
            np.sum(np.diag(np.dot(self.v_solver.model.Z.T, self.v_solver.model.Z)) * (vs_curr**2 + sigma2_v_expect)) -
            2 * np.dot(data.y, np.dot(self.v_solver.model.Z, vs_curr))
        ) / len(data.y) * np.ones(len(data.y))

        data.y = data.obs + self.v_solver.model.forward(vs_curr)
