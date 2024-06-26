import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass

from anml.parameter.parameter import Parameter
from anml.parameter.prior import Prior
from anml.parameter.processors import process_all
from anml.parameter.spline_variable import Spline, SplineLinearConstr
from anml.parameter.variables import Intercept
from sfma.data import DataSpecs
from sfma.data import Data
from sfma.models.marginal import MarginalModel
from anml.solvers.base import ScipyOpt
from anml.solvers.composite import TrimmingSolver

from matplotlib import pyplot as plt


@dataclass
class InputVar:
    """
    An input variable for frontier, with derivative constraints and spline specs.
    """

    col_input: str
    increasing: bool = False
    decreasing: bool = False
    concave: bool = False
    convex: bool = False
    knots_type: str = 'domain'
    knots_num: int = 4
    knots_degree: int = 3
    r_linear: bool = False
    l_linear: bool = False
    constr_grid_num: int = 20

    def __post_init__(self):
        if self.concave and self.convex:
            raise RuntimeError("Cannot enforce both concavity and convexity.")
        if self.increasing and self.decreasing:
            raise RuntimeError("Cannot enforce both increasing and decreasing trend.")
        if self.knots_type not in ['domain', 'frequency']:
            raise RuntimeError("Knots type needs to be either domain or frequency")
        if self.knots_num < 2:
            raise RuntimeError("Need at least two knots.")
        if self.knots_degree < 1:
            raise RuntimeError("Need at least degree one.")


class SFMAModel:
    def __init__(self,
                 df: pd.DataFrame,
                 col_output: str,
                 col_se: str,
                 covariates: List[InputVar],
                 include_intercept: bool = True,
                 include_gamma: bool = False,
                 pct_trimming: float = 0.0):

        self.col_output = col_output
        self.col_se = col_se
        self.covariates = covariates

        self.include_intercept = include_intercept
        self.include_gamma = include_gamma
        self.pct_trimming = pct_trimming

        if pct_trimming >= 1.0 or pct_trimming < 0.0:
            raise RuntimeError("Need to have pct trimming < 1.0 and >= 0.")

        data = df.copy().reset_index(drop=True)
        data['group'] = data.index

        self.data_spec = DataSpecs(col_obs=self.col_output, col_obs_se=self.col_se)

        params = [
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

        # Create derivative constraints including monotone constraints
        # and shape constraints
        derivative_constraints = []
        for covariate in self.covariates:
            if covariate.increasing or covariate.decreasing:
                if covariate.increasing:
                    y_bounds = [0.0, np.inf]
                else:
                    y_bounds = [-np.inf, 0.0]
                derivative_constraints.append(
                    SplineLinearConstr(order=1, y_bounds=y_bounds, grid_size=covariate.constr_grid_num)
                )
            if covariate.convex or covariate.concave:
                if covariate.convex:
                    y_bounds = [0.0, np.inf]
                else:
                    y_bounds = [-np.inf, 0.0]
                derivative_constraints.append(
                    SplineLinearConstr(order=2, y_bounds=y_bounds, grid_size=covariate.constr_grid_num)
                )

            # Create the spline variable for the input
            params.append([
                Parameter(
                    param_name=f"beta_{covariate.col_input}",
                    variables=[
                        Spline(
                            covariate=covariate.col_input,
                            knots_type=covariate.knots_type,
                            knots_num=covariate.knots_num,
                            degree=covariate.knots_degree,
                            r_linear=covariate.r_linear,
                            l_linear=covariate.l_linear,
                            derivative_constr=derivative_constraints,
                            include_intercept=True
                        )
                    ]
                )
            ])

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
        self.trimming = TrimmingSolver([self.solver])

        # Initialize parameter values
        self.x_init = self.marginal_model.get_var_init(self.data)

        # Placeholder for inefficiencies
        self.inefficiencies = np.zeros(self.data.num_obs)

    def fit(self, **kwargs):
        x_init = kwargs.pop('x_init', self.x_init)
        options = kwargs.pop('options', {'solver_options': {}})
        self.trimming.fit(x_init=x_init, data=self.data, n=len(self.data.obs),
                          pct_trimming=self.pct_trimming, options=options, **kwargs)
        self.inefficiencies = self.marginal_model.get_ie(self.solver.x_opt, self.data)
        if self.solver.success:
            print("\nModel converged successfully! :)")
        else:
            print("\nModel failed to converge! :(")

    def predict(self, df):
        """
        Create predictions based on a (potentially new) data frame.

        :param df: pd.DataFrame
        """
        mats = []
        for covariate in self.covariates:
            mat = self.marginal_model.params[f"beta_{covariate.col_input}"].variables[0]._design_matrix(
                df[covariate.col_input],
                create_spline=False
            )
            mats.append(mat)
        d_mat = np.hstack(mats)
        predictions = self.solver.predict(mat=d_mat)
        return predictions

    # TODO: Need to discuss how to visualize multiple variables -- this won't work in its current form
    def plot_frontier(self, df, true_frontier=None):
        """
        Plot a frontier estimate with the data.

        :param df: data frame to make predictions from
        :param true_frontier: np.ndarray where the first column is x's and second is y's
        """
        predictions = self.predict(df)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        if true_frontier is not None:
            ax.plot(true_frontier[:, 0], true_frontier[:, 1], linestyle='solid', c='black',
                    label='truth')
        ax.scatter(df[self.col_input], df[self.col_output], c='grey', alpha=0.5, label='observed')
        ax.errorbar(df[self.col_input],
                    df[self.col_output],
                    yerr=df[self.col_se],
                    linestyle='None', elinewidth=0.5, c='grey', alpha=0.5)
        ax.plot(df[self.col_input], predictions, c='blue', label='predicted')
        ax.set_xlabel(self.col_input)
        ax.set_ylabel(self.col_output)

        if self.pct_trimming > 0.0:
            outliers = self.marginal_model.w == 0
            outlier_x = df[self.col_input][outliers]
            outlier_y = df[self.col_output][outliers]
            ax.scatter(outlier_x, outlier_y, c='red', marker="X", label='trimmed')

        ax.legend()

    def plot_inefficiencies(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(self.inefficiencies)


