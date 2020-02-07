# -*- coding: utf-8 -*-
"""
    test_data
    ~~~~~~~~~
    Test `data` model for `sfma` package.
"""
import numpy as np
import pandas as pd
import pytest
import sfma.data as data


@pytest.fixture()
def test_df():
    num_obs = 5
    df = pd.DataFrame({
        'obs': np.random.randn(num_obs),
        'obs_se': np.random.rand(num_obs) + 0.01,
        'cov0': np.random.randn(num_obs),
        'cov1': np.random.randn(num_obs),
        'cov2': np.random.randn(num_obs),
    })
    return df


@pytest.mark.parametrize('obs', ['obs', None])
@pytest.mark.parametrize('obs_se', ['obs_se', None])
def test_obs(test_df, obs, obs_se):
    d = data.Data(test_df,
                  col_obs=obs,
                  col_obs_se=obs_se,
                  col_covs=['cov0', 'cov1', 'cov2'])
    if obs is None:
        assert d.obs.shape == (test_df.shape[0], 0)
    else:
        assert d.obs.shape == (test_df.shape[0],)

    if obs_se is None:
        assert d.obs_se.shape == (test_df.shape[0], 0)
    else:
        assert d.obs_se.shape == (test_df.shape[0],)

    assert d.num_obs == test_df.shape[0]


@pytest.mark.parametrize('covs', [None,
                                  ['cov0', 'cov1', 'cov2']])
@pytest.mark.parametrize('add_intercept', [True, False])
def test_covs(test_df, covs, add_intercept):
    d = data.Data(test_df,
                  col_obs='obs',
                  col_obs_se='obs_se',
                  col_covs=covs,
                  add_intercept=add_intercept)

    num_covs = 0 if covs is None else len(covs)
    num_covs += add_intercept
    print(d.col_covs)
    assert d.covs.shape == (d.num_obs, num_covs)


@pytest.mark.parametrize('u_id', [None,
                                  np.array([0, 0, 1, 1, 2])])
def test_u_id(test_df, u_id):
    if u_id is not None:
        test_df['u_id'] = u_id
        u_id = 'u_id'
    d = data.Data(test_df,
                  col_obs='obs',
                  col_obs_se='obs_se',
                  col_covs=['cov0', 'cov1', 'cov2'],
                  col_u_id=u_id)

    if u_id is None:
        assert np.allclose(d.u_id, np.arange(d.num_obs))
        assert d.num_u_groups == d.num_obs
        assert np.allclose(np.sort(d.u_group_sizes.index.values),
                           np.arange(d.num_obs))
        assert np.allclose(d.u_group_sizes, 1.0)
        assert all([
            np.allclose(d.u_group_idx[i], np.array([i]))
            for i in d.u_group_sizes.index
        ])
    else:
        assert np.allclose(d.u_id, np.array([0, 0, 1, 1, 2]))
        assert d.num_u_groups == 3
        assert np.allclose(np.sort(d.u_group_sizes.index.values),
                           np.array([0, 1, 2]))
        assert np.allclose(d.u_group_sizes, np.array([2, 2, 1]))
        assert all([
            np.allclose(d.u_group_idx[0], np.array([0, 1])),
            np.allclose(d.u_group_idx[1], np.array([2, 3])),
            np.allclose(d.u_group_idx[2], np.array([4])),
        ])


@pytest.mark.parametrize('v_id', [None,
                                  np.array([0, 0, 1, 1, 2])])
def test_v_id(test_df, v_id):
    if v_id is not None:
        test_df['v_id'] = v_id
        v_id = 'v_id'
    test_df['u_id'] = np.arange(test_df.shape[0])[::-1]
    d = data.Data(test_df,
                  col_obs='obs',
                  col_obs_se='obs_se',
                  col_covs=['cov0', 'cov1', 'cov2'],
                  col_u_id='u_id',
                  col_v_id=v_id)
    d.sort_by_u_id()
    if v_id is None:
        assert np.allclose(d.v_id, np.arange(d.num_obs)[::-1])
        assert d.num_v_groups == d.num_obs
        assert np.allclose(np.sort(d.v_group_sizes.index.values),
                           np.arange(d.num_obs))
        assert np.allclose(d.v_group_sizes, 1.0)
        assert all([
            np.allclose(d.v_group_idx[i], d.num_obs - np.array([i]) - 1)
            for i in d.v_group_sizes.index
        ])
    else:
        assert np.allclose(d.v_id, np.array([2, 1, 1, 0, 0]))
        assert d.num_v_groups == 3
        assert np.allclose(np.sort(d.v_group_sizes.index.values),
                           np.array([0, 1, 2]))
        assert np.allclose(d.v_group_sizes, np.array([2, 2, 1]))
        assert all([
            np.allclose(d.v_group_idx[0], np.array([3, 4])),
            np.allclose(d.v_group_idx[1], np.array([1, 2])),
            np.allclose(d.v_group_idx[2], np.array([0])),
        ])


@pytest.mark.parametrize('u_id', [np.arange(5)])
@pytest.mark.parametrize('v_id', [np.arange(5)[::-1]])
def test_sort_by(test_df, u_id, v_id):
    test_df['u_id'] = u_id
    test_df['v_id'] = v_id
    d = data.Data(test_df,
                  col_obs='obs',
                  col_obs_se='obs_se',
                  col_covs=['cov0', 'cov1', 'cov2'],
                  col_u_id='u_id',
                  col_v_id='v_id')

    d.sort_by_u_id()
    assert np.allclose(d.u_id, np.arange(d.num_obs))
    assert np.allclose(d.v_id, np.arange(d.num_obs)[::-1])

    d.sort_by_v_id()
    assert np.allclose(d.u_id, np.arange(d.num_obs)[::-1])
    assert np.allclose(d.v_id, np.arange(d.num_obs))
