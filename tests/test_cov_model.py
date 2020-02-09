# -*- coding: utf-8 -*-
"""
    test_cov_model
    ~~~~~~~~~~~~~~
    Test `cov_model` module for `sfma` package.
"""
import numpy as np
import pandas as pd
import pytest
from sfma import CovModel
from sfma import Data


@pytest.fixture()
def data():
    num_obs = 5
    df = pd.DataFrame({
        'obs': np.random.randn(num_obs),
        'obs_se': np.repeat(0.1, num_obs),
        'cov0': np.random.randn(num_obs),
        'cov1': np.random.rand(num_obs),
        'u_id': np.array([0, 1, 1, 2, 2]),
        'v_id': np.arange(num_obs)
    })

    return Data(df,
                col_obs='obs',
                col_obs_se='obs_se',
                col_covs=['cov0', 'cov1'],
                col_u_id='u_id',
                col_v_id='v_id',
                add_intercept=True)


@pytest.mark.parametrize('cov_name', ['intercept', 'cov0', 'cov1'])
def test_default_input(cov_name):
    cov_model = CovModel(cov_name)
    cov_model.check_attr()


@pytest.mark.parametrize('add_u', [True, False])
def test_add_u(add_u):
    cov_model = CovModel('intercept', add_u=add_u)
    assert add_u == cov_model.num_random_vars


@pytest.mark.parametrize('add_spline', [True, False])
def test_add_spline(add_spline):
    cov_model = CovModel('cov0', add_spline=add_spline)
    if add_spline:
        assert cov_model.num_fixed_vars == cov_model.spline_knots.size + \
            cov_model.spline_degree - 2
    else:
        assert cov_model.num_fixed_vars == 1


@pytest.mark.parametrize('spline_knots', [None, np.linspace(0.1, 0.9, 9)])
def test_spline_konts(spline_knots):
    cov_model = CovModel('cov0', add_spline=True)
    if spline_knots is not None:
        cov_model.update_attr(spline_knots=spline_knots)
    assert cov_model.spline_knots[0] == 0.0
    assert cov_model.spline_knots[-1] == 1.0


@pytest.mark.parametrize('spline_monotonicity', [None,
                                                 'increasing',
                                                 'decreasing'])
@pytest.mark.parametrize('spline_convexity', [None,
                                              'convex',
                                              'concave'])
def test_spline_shape_constraints(spline_monotonicity, spline_convexity):
    cov_model = CovModel('cov0',
                         spline_monotonicity=spline_monotonicity,
                         spline_convexity=spline_convexity)
    assert cov_model.num_constraints == 0
    cov_model.update_attr(add_spline=True)
    cov_model.update_attr(spline_num_constraint_points=10)

    assert cov_model.num_constraints == 10*(
        (spline_monotonicity is not None) +
        (spline_convexity is not None)
    )


@pytest.mark.parametrize('spline_knots_type', ['frequency', 'domain'])
@pytest.mark.parametrize('spline_knots', [np.linspace(0.0, 1.0, 4)])
@pytest.mark.parametrize('spline_degree', [4])
@pytest.mark.parametrize('spline_l_linear', [True, False])
@pytest.mark.parametrize('spline_r_linear', [True, False])
def test_create_spline(data,
                       spline_knots_type,
                       spline_knots,
                       spline_degree,
                       spline_l_linear,
                       spline_r_linear):
    cov_model = CovModel('cov0',
                         spline_knots_type=spline_knots_type,
                         spline_knots=spline_knots,
                         spline_degree=spline_degree,
                         spline_l_linear=spline_l_linear,
                         spline_r_linear=spline_r_linear)
    spline = cov_model.create_spline(data)

    if spline_knots_type == 'frequency':
        assert np.allclose(spline.knots,
                           np.quantile(data.covs[cov_model.cov_name],
                                       cov_model.spline_knots))
    else:
        assert np.allclose(spline.knots,
                           data.covs[cov_model.cov_name].min() +
                           cov_model.spline_knots*(
                               data.covs[cov_model.cov_name].max() -
                               data.covs[cov_model.cov_name].min()
                           ))

    assert cov_model.num_fixed_vars == 1
    cov_model.update_attr(add_spline=True)
    assert cov_model.num_fixed_vars == spline.num_spline_bases - 1


@pytest.mark.parametrize('add_spline', [True, False])
def test_design_mat(data, add_spline):
    cov_model = CovModel('cov0', add_spline=add_spline)
    mat = cov_model.create_design_mat(data)
    spline = cov_model.create_spline(data)
    if add_spline:
        assert np.allclose(mat,
                           spline.design_mat(
                               data.covs[cov_model.cov_name]
                           )[:, 1:])
    else:
        assert np.allclose(mat.ravel(), data.covs[cov_model.cov_name])


@pytest.mark.parametrize('add_spline', [True, False])
@pytest.mark.parametrize('spline_monotonicity', [None,
                                                 'increasing',
                                                 'decreasing'])
@pytest.mark.parametrize('spline_convexity', [None,
                                              'convex',
                                              'concave'])
@pytest.mark.parametrize('spline_num_constraint_points', [10, 20])
def test_constraint_mat(data, add_spline,
                        spline_monotonicity,
                        spline_convexity,
                        spline_num_constraint_points):
    cov_model = CovModel(
        'cov0',
         add_spline=add_spline,
         spline_monotonicity=spline_monotonicity,
         spline_convexity=spline_convexity,
         spline_num_constraint_points=spline_num_constraint_points
    )
    mat = cov_model.create_constraint_mat(data)
    assert mat.shape == (cov_model.num_constraints, cov_model.num_fixed_vars)
