# -*- coding: utf-8 -*-
"""
    model
    ~~~~~
    `model` module of the `sfma` package.
"""
import numpy as np
import xspline


class CovModel:
    """Covariates model.
    """
    def __init__(self, cov_name,
                 add_spline=False,
                 add_u=False,
                 spline_knots_type='frequency',
                 spline_knots=np.array([0.0, 1.0]),
                 spline_degree=3,
                 spline_l_linear=False,
                 spline_r_linear=False,
                 spline_monotonicity=None,
                 spline_convexity=None,
                 spline_num_constraint_points=20):
        """Constructor of the CovModel.

        Args:
            cov_name(str):
                Corresponding covariate column name in the data frame.
        Keyword Args:
            add_spline (bool, optional):
                If using spline.
            add_u (bool, optional):
                If add random for this covariate.
            spline_knots_type (str, optional):
                The method of how to place the knots, `'frequency'` place the
                knots according to the data quantile and `'domain'` place the
                knots according to the domain of the data.
            spline_knots (np.ndarray, optional):
                A numpy array between 0 and 1 contains the relative position of
                the knots placement, with respect to either frequency or domain.
            spline_degree (int, optional):
                The degree of the spline.
            spline_l_linear (bool, optional):
                If use left linear tail.
            spline_r_linear (bool, optional):
                If use right linear tail.
            spline_monotonicity (str | None, optional):
                Spline shape prior, `'increasing'` indicates spline is
                increasing, `'decreasing'` indicates spline is decreasing.
            spline_convexity (str | None, optional):
                Spline shape prior, `'convex'` indicate if spline is convex and
                `'concave'` indicate spline is concave.
            spline_num_constraint_points(int, optional):
                Number of constraint points used for approximation.
        """
        # check input

        self.cov_name = cov_name
        self.add_spline = add_spline
        self.add_u = add_u

        self.spline_knots_type = spline_knots_type
        self.spline_knots = spline_knots
        self.spline_degree = spline_degree
        self.spline_l_linear = spline_l_linear
        self.spline_r_linear = spline_r_linear
        self.spline_monotonicity = spline_monotonicity
        self.spline_convexity = spline_convexity
        self.spline_num_constraint_points = spline_num_constraint_points
        self.check_attr()

    def check_attr(self):
        """Check spline parameters.
        """
        assert isinstance(self.cov_name, str)
        assert isinstance(self.add_spline, bool)
        assert isinstance(self.add_u, bool)
        assert isinstance(self.add_spline, bool)
        assert self.spline_knots_type in ['frequency', 'domain']
        assert isinstance(self.spline_knots, np.ndarray)
        assert np.min(self.spline_knots) >= 0.0
        assert np.max(self.spline_knots) <= 1.0
        assert isinstance(self.spline_degree, int)
        assert self.spline_degree >= 0
        assert isinstance(self.spline_l_linear, bool)
        assert isinstance(self.spline_r_linear, bool)
        assert self.spline_monotonicity in ['increasing', 'decreasing'] or \
            self.spline_monotonicity is None
        assert self.spline_convexity in ['convex', 'concave'] or \
            self.spline_convexity is None
        assert isinstance(self.spline_num_constraint_points, int)
        assert self.spline_num_constraint_points > 0

        # construct the knots
        self.spline_knots = np.unique(self.spline_knots)
        if np.min(self.spline_knots) > 0.0:
            self.spline_knots = np.insert(self.spline_knots, 0, 0.0)
        if np.max(self.spline_knots) < 1.0:
            self.spline_knots = np.append(self.spline_knots, 1.0)

    def update_attr(self, **kwargs):
        """Refresh all the attributes of the class.

        Args:
            **kwargs:
                Keyword arguments related to the spline attributes.
        """
        assert all([hasattr(self, key) for key in kwargs.keys()])
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        self.check_attr()

    @property
    def num_fixed_vars(self):
        if self.add_spline:
            n = self.spline_knots.size - \
                self.spline_l_linear - self.spline_r_linear + \
                self.spline_degree - 2
        else:
            n = 1
        return n

    @property
    def num_random_vars(self):
        return self.num_fixed_vars if self.add_u else 0

    @property
    def num_constraints(self):
        if not self.add_spline:
            return 0
        else:
            return self.spline_num_constraint_points*(
                (self.spline_monotonicity is not None) +
                (self.spline_convexity is not None)
            )

    def create_spline(self, data):
        """Create spline given current spline parameters.

        Args:
            data (sfma.Data):
                The data frame used for storing the data
        Returns:
            xspline.XSpline
                The spline object.
        """
        # extract covariate
        assert self.cov_name in data.covs.columns
        cov = data.covs[self.cov_name]

        if self.spline_knots_type == 'frequency':
            spline_knots = np.quantile(cov, self.spline_knots)
        else:
            spline_knots = cov.min() + self.spline_knots*(cov.max() - cov.min())

        return xspline.XSpline(spline_knots,
                               self.spline_degree,
                               l_linear=self.spline_l_linear,
                               r_linear=self.spline_r_linear)

    def create_design_mat(self, data):
        """Create design matrix.

        Args:
            data (sfma.Data):
                The data frame used for storing the data

        Returns:
            numpy.ndarray:
                Return the design matrix for linear cov or spline.
        """
        assert self.cov_name in data.covs.columns
        cov = data.covs[self.cov_name].values
        if self.add_spline:
            spline = self.create_spline(data)
            mat = spline.design_mat(cov)[:, 1:]
        else:
            mat = cov[:, None]

        return mat

    def create_constraint_mat(self, data):
        """Create constraints matrix.

        Args:
            data (sfma.Data):
                The data frame used for storing the data

        Returns:
            numpy.ndarray:
                Return constraints matrix if have any.
        """
        assert self.cov_name in data.covs.columns
        cov = data.covs[self.cov_name].values

        mat = np.array([]).reshape(0, self.num_fixed_vars)
        if not self.add_spline:
            return mat

        points = np.linspace(cov.min(), cov.max(),
                             self.spline_num_constraint_points)
        spline = self.create_spline(data)

        if self.spline_monotonicity is not None:
            sign = 1.0 if self.spline_monotonicity is 'decreasing' else -1.0
            mat = np.vstack((mat,
                             sign*spline.design_dmat(points, 1)[:, 1:]))

        if self.spline_convexity is not None:
            sign = 1.0 if self.spline_convexity is 'concave' else -1.0
            mat = np.vstack((mat,
                             sign*spline.design_dmat(points, 2)[:, 1:]))

        return mat
