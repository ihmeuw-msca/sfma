import numpy as np
import pandas as pd

from anml.parameter.parameter import Parameter
from anml.parameter.prior import Prior
from anml.parameter.processors import process_all
from anml.parameter.spline_variable import Spline, SplineLinearConstr
from anml.parameter.variables import Intercept
from sfma.data import DataSpecs
from sfma.data import Data
from sfma.models.marginal import MarginalModel
from anml.solvers.base import ScipyOpt


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
                 include_intercept: bool = True,
                 include_gamma: bool = False):

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
        self.include_gamma = include_gamma

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
        params = [
            Parameter(
                param_name="beta",
                variables=[
                    Spline(
                        covariate=self.col_input,
                        knots_type=self.knots_type,
                        knots_num=self.knots_num,
                        degree=self.knots_degree,
                        r_linear=self.r_linear,
                        l_linear=self.l_linear,
                        derivative_constr=derivative_constraints,
                        include_intercept=True
                    )
                ]
            ),
            Parameter(
                param_name="gamma",
                variables=[
                    Intercept(
                        fe_prior=Prior(
                            lower_bound=[0.0],
                            upper_bound=[np.inf if self.include_gamma else 0.0]
                        )
                    )
                ]
            ),
            Parameter(
                param_name="eta",
                variables=[
                    Intercept(
                        fe_prior=Prior(lower_bound=[0.0], upper_bound=[np.inf])
                    )
                ]
            )
        ]

        # Create the parameter set to solve
        # the marginal likelihood problem
        for param in params:
            process_all(param, data)

        # Process the data set
        self.data = Data(data_specs=self.data_spec)
        self.data.process(data)

        # Build the two models
        self.marginal_model = MarginalModel(params)

        # Create the solvers for the models
        self.solver = ScipyOpt(self.marginal_model)

        # Initialize parameter values
        self.x_init = self.marginal_model.get_var_init(self.data)

        # Placeholder for inefficiencies
        self.inefficiencies = np.zeros(self.data.num_obs)

    def fit(self, **kwargs):
        self.solver.fit(x_init=self.x_init, data=self.data, **kwargs)
        self.inefficiencies = self.marginal_model.get_ie(self.solver.x_opt, self.data)

    def predict(self):
        return self.solver.predict()
