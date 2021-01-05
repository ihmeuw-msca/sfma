import numpy as np
import pandas as pd

from anml.parameter.parameter import Parameter, ParameterSet
from anml.parameter.prior import GaussianPrior, Prior
from anml.parameter.processors import process_all
from anml.parameter.spline_variable import Spline, SplineLinearConstr
from anml.parameter.variables import Intercept
from sfma.data import DataSpecs
from sfma.data import Data
from sfma.models.maximal import VModel, UModel
from sfma.models.marginal import SimpleBetaEtaModel
from anml.solvers.base import ClosedFormSolver, ScipyOpt, IPOPTSolver
from sfma.solver import SimpleSolver


class SFMAModel:
    def __init__(self,
                 df: pd.DataFrame,
                 col_output: str,
                 col_se: str,
                 col_input: str,
                 increasing: bool = False,
                 decreasing: bool = False,
                 concave: bool = False,
                 convex: bool = False,
                 knots_type: str = 'domain',
                 knots_num: int = 4,
                 knots_degree: int = 3,
                 r_linear: bool = False,
                 l_linear: bool = False,
                 include_intercept: bool = True):

        self.col_output = col_output
        self.col_se = col_se
        self.col_input = col_input
        self.increasing = increasing
        self.decreasing = decreasing
        self.concave = concave
        self.convex = convex
        self.knots_num = knots_num
        self.knots_type = knots_type
        self.knots_degree = knots_degree
        self.r_linear = r_linear
        self.l_linear = l_linear
        self.include_intercept = include_intercept

        data = df.copy().reset_index(drop=True)
        data['group'] = data.index

        self.data_spec = DataSpecs(col_obs=self.col_output, col_obs_se=self.col_se)

        # Create derivative constraints including monotone constraints
        # and shape constraints
        derivative_constraints = []
        if self.increasing or self.decreasing:
            if self.increasing and self.decreasing:
                raise RuntimeError("Cannot have both monotone increasing and decreasing.")
            if self.increasing:
                y_bounds = [0.0, np.inf]
            else:
                y_bounds = [-np.inf, 0.0]
            derivative_constraints.append(
                SplineLinearConstr(order=1, y_bounds=y_bounds, grid_size=20)
            )
        if self.convex or self.concave:
            if self.convex and self.concave:
                raise RuntimeError("Cannot have both convex and concave.")
            if self.convex:
                y_bounds = [0.0, np.inf]
            else:
                y_bounds = [-np.inf, 0.0]
            derivative_constraints.append(
                SplineLinearConstr(order=2, y_bounds=y_bounds, grid_size=20)
            )

        # Create the spline variable for the input
        spline = Spline(
            covariate=self.col_input,
            knots_type=self.knots_type,
            knots_num=self.knots_num,
            degree=self.knots_degree,
            r_linear=self.r_linear,
            l_linear=self.l_linear,
            derivative_constr=derivative_constraints,
            include_intercept=include_intercept
        )

        # Create the parameter set to solve
        # the marginal likelihood problem
        param_set_marginal = ParameterSet(
            parameters=[
                Parameter(
                    param_name='beta',
                    variables=[spline]
                )
            ]
        )
        process_all(param_set_marginal, data)

        # Create the parameter set to solve
        # for the inefficiencies
        intercept = Intercept(
            add_re=True, col_group='group',
            re_prior=GaussianPrior(
                lower_bound=[0.0],
                upper_bound=[np.inf]
            )
        )
        param_set_v = ParameterSet(
            parameters=[
                Parameter(
                    param_name='v',
                    variables=[intercept]
                )
            ]
        )
        process_all(param_set_v, data)

        # Process the data set
        self.data = Data(data_specs=self.data_spec)
        self.data.process(data)

        # Build the two models
        self.marginal_model = SimpleBetaEtaModel(param_set_marginal)
        self.v_model = VModel(param_set_v)

        # Create the solvers for the models
        marginal_solver = IPOPTSolver(self.marginal_model)
        v_solver = ClosedFormSolver(self.v_model)
        self.solver = SimpleSolver([marginal_solver, v_solver])

        # Randomly initialize parameter values
        self.x_init = np.random.randn(marginal_solver.model.x_dim)

        # Placeholder for inefficiencies
        self.inefficiencies = np.zeros(len(data))

    def fit(self):
        self.solver.fit(x_init=self.x_init, data=self.data)
        self.inefficiencies = self.solver.solvers[1].x_opt

    def predict(self):
        return self.solver.predict()

    @property
    def marginal_result(self):
        return self.solver.solvers[0].info['status_msg']

    @property
    def v_result(self):
        return self.solver.solvers[1].info['status_msg']

    @property
    def inefficiencies(self):
        return self._inefficiencies

    @inefficiencies.setter
    def inefficiencies(self, inefficiencies: np.ndarray):
        self._inefficiencies = inefficiencies
