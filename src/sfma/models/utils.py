import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from typing import List, Tuple

from anml.parameter.utils import combine_constraints


def build_linear_constraint(constraints: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    if len(constraints) == 1:
        mats, lbs, ubs = constraints[0]
    mats, lbs, ubs = zip(*constraints)
    C, c_lb, c_ub = combine_constraints(mats, lbs, ubs)
    if np.count_nonzero(C) == 0:
        return None, None, None 
    else:
        return C, c_lb, c_ub
