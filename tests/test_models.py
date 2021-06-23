import numpy as np
import pytest

import scipy.stats as stats
from scipy.stats import halfnorm
from scipy.stats import expon
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
from anml.solvers.composite import TrimmingSolver


class Simulator:
    def __init__(self, nu: int, gamma: float, sigma_min: float, sigma_max: float,
                 x: callable, func: callable, ineff_dist: str = 'half-normal'):
        """
        Simulation class for stochastic frontier meta-analysis.

        nu
            The scale of the inefficiency term
        gamma
            The variance of the random effect term
        sigma_min, sigma_max
            The study-specific errors, max and minimum. They will be drawn from a uniform distribution.
        x
            A callable function to generate a realization from a random variable x (is the covariate used
            to construct the frontier). Needs to have an argument size.
        func
            A function of x that defines the frontier
        ineff_dist
            Inefficiency distribution
        """
        self.nu = nu
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.x = x
        self.func = func

        if ineff_dist == 'half-normal':
            self.rvs = halfnorm.rvs
        elif ineff_dist == 'exponential':
            self.rvs = expon.rvs
        else:
            raise RuntimeError("Inefficiency distribution must be half-normal or exponential")

    def simulate(self, n: int = 1, **kwargs):
        np.random.seed(365)
        sigma = stats.uniform.rvs(loc=self.sigma_min, scale=self.sigma_max, size=n)
        epsilon = stats.norm.rvs(loc=0, scale=sigma, size=n)

        us = stats.norm.rvs(loc=0, scale=self.gamma, size=n)
        vs = self.rvs(scale=self.nu, size=n)

        xs = self.x(size=n, **kwargs)
        front = self.func(xs)
        observed = front + us - vs + epsilon
        return us, vs, epsilon, sigma, xs, front, observed


@pytest.fixture
def settings():
    return {
        'col_output': 'output',
        'col_se': 'se',
        'col_input': 'input',
        'knots_num': 3,
        'knots_type': 'frequency',
        'knots_degree': 3,
        'include_gamma': True
    }


@pytest.fixture
def df(settings):
    np.random.seed(16)
    s = Simulator(nu=1, gamma=0.25, sigma_min=0, sigma_max=0.75,
                  x=lambda size: stats.uniform.rvs(size=size, loc=0.5), func=lambda x: np.log(x) + 10)
    us, vs, epsilon, sigma, xs, front, observed = s.simulate(n=30)
    return pd.DataFrame({
        settings['col_input']: xs,
        settings['col_output']: observed,
        settings['col_se']: sigma
    })


@pytest.fixture
def params(settings):
    params = [
        Parameter(
            param_name="beta",
            variables=[
                Spline(
                    covariate=settings['col_input'],
                    knots_type=settings['knots_type'],
                    knots_num=settings['knots_num'],
                    degree=settings['knots_degree'],
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
                        upper_bound=[np.inf if settings['include_gamma'] else 0.0]
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
    return params


def test_model(df, params, settings):

    df2 = df.copy().reset_index(drop=True)
    df2['group'] = df2.index

    data_spec = DataSpecs(col_obs=settings['col_output'], col_obs_se=settings['col_se'])

    for param in params:
        process_all(param, df)

    # Process the data set
    data = Data(data_specs=data_spec)
    data.process(df)

    marginal_model = MarginalModel(params)
    solver = ScipyOpt(marginal_model)

    # Initialize parameter values
    x_init = marginal_model.get_var_init(data)
    np.random.seed(10)
    solver.fit(x_init=x_init, data=data, options={'solver_options': {},
                                                  'tol': 1e-16})
    inefficiencies = marginal_model.get_ie(solver.x_opt, data)
    assert solver.x_opt.size == 7
    assert inefficiencies.size == 30
    assert all(inefficiencies > 0)


def test_model_trimming(df, params, settings):

    df2 = df.copy().reset_index(drop=True)
    df2['group'] = df2.index

    data_spec = DataSpecs(col_obs=settings['col_output'], col_obs_se=settings['col_se'])

    for param in params:
        process_all(param, df)

    # Process the data set
    data = Data(data_specs=data_spec)
    data.process(df)

    marginal_model = MarginalModel(params)
    solver = ScipyOpt(marginal_model)
    trimming = TrimmingSolver([solver])

    # Initialize parameter values
    x_init = marginal_model.get_var_init(data)

    trimming.fit(x_init=x_init, data=data, options={'solver_options': {},
                                                    'tol': 1e-16},
                 n=len(df), pct_trimming=0.0)
    inefficiencies = marginal_model.get_ie(trimming.solvers[0].x_opt, data)
    assert trimming.solvers[0].x_opt.size == 7
    assert inefficiencies.size == 30
    assert all(inefficiencies > 0)

    solver = ScipyOpt(marginal_model)
    trimming = TrimmingSolver([solver])
    trimming.fit(x_init=x_init, data=data, options={'solver_options': {},
                                                    'tol': 1e-16},
                 n=len(df), pct_trimming=0.3)
    inefficiencies = marginal_model.get_ie(trimming.solvers[0].x_opt, data)
    assert trimming.solvers[0].x_opt.size == 7
    assert inefficiencies.size == 30
