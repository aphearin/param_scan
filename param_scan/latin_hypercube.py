"""Use scipy.stats.qmc to generate latin hypercube samples."""
import numpy as np

shape_errmsg = "param_bounds must have shape (num_params, 2)"
minmax_errmsg = "All (min, max) entries of param_bounds must have min < max"


def latin_hypercube_pydoe(param_bounds, num_evaluations, seed=None):
    """Generate a latin hypercube oriented with the Cartesian axes.

    Parameters
    ----------
    param_bounds : n_dim-length sequence of 2-element tuples
        Each entry of param_bounds (min, max) specifies the bounds in each dimension

    num_evaluations : int
        Number of points in sample

    Returns
    -------
    sample : ndarray, shape(num_evaluations, n_dim)
        Latin hypercube centered on zero

    """
    from pyDOE2 import lhs

    param_bounds = np.atleast_2d(param_bounds)
    assert param_bounds.shape[1] == 2, shape_errmsg
    _dx = np.diff(param_bounds, axis=1)
    assert np.all(_dx > 0), minmax_errmsg
    num_params = param_bounds.shape[0]

    rng = np.random.RandomState(seed)
    unit_hypercube = lhs(num_params, samples=num_evaluations, random_state=rng)
    xmins = param_bounds[:, 0]
    xmaxs = param_bounds[:, 1]

    params = np.zeros_like(unit_hypercube)
    for i in range(num_params):
        xmin, xmax = xmins[i], xmaxs[i]
        params[:, i] = xmin + (xmax - xmin) * unit_hypercube[:, i]
    return params


def latin_hypercube(param_bounds, num_evaluations, seed=None):
    """Generate a latin hypercube oriented with the Cartesian axes.

    Parameters
    ----------
    param_bounds : n_dim-length sequence of 2-element tuples
        Each entry of param_bounds (min, max) specifies the bounds in each dimension

    num_evaluations : int
        Number of points in sample

    seed : int, optional
        Random number seed

    Returns
    -------
    sample : ndarray, shape(num_evaluations, n_dim)
        Latin hypercube centered on zero

    """
    from scipy.stats.qmc import LatinHypercube

    param_bounds = np.atleast_2d(param_bounds)
    assert param_bounds.shape[1] == 2, shape_errmsg
    _dx = np.diff(param_bounds, axis=1)
    assert np.all(_dx > 0), minmax_errmsg
    num_params = param_bounds.shape[0]

    LH = LatinHypercube(num_params, seed=seed)
    unit_hypercube = LH.random(num_evaluations)

    xmins = param_bounds[:, 0]
    xmaxs = param_bounds[:, 1]

    params = np.zeros_like(unit_hypercube)
    for i in range(num_params):
        xmin, xmax = xmins[i], xmaxs[i]
        params[:, i] = xmin + (xmax - xmin) * unit_hypercube[:, i]
    return params


def _get_eigenbasis_transform(cov):
    """X_orig = X_espace.dot(T)"""
    evals, V = np.linalg.eig(cov)
    R, S = V, np.sqrt(np.diag(evals))
    T = R.dot(S).T
    return T


def latin_hypercube_from_cov(mu, cov, sig, num_evaluations):
    """Generate a latin hypercube that encompasses some multivariate Gaussian data.

    Parameters
    ----------
    mu : ndarray, shape (n_dim, )

    cov : ndarray, shape (n_dim, n_dim)

    sig : float or ndarray of shape (n_dim, )
        Number of sigma used to define the box length

    num_evaluations : int
        Number of points in sample

    Returns
    -------
    sample : ndarray, shape(num_evaluations, n_dim)
        Latin hypercube centered on mu rotated in the eigenbasis defined by cov

    """
    n_dim = mu.size

    _sig = np.atleast_1d(sig)
    assert np.all(_sig > 0), "Input sig must be strictly positive"
    if len(_sig) == 1:
        param_bounds = [(-sig, sig)] * n_dim
    else:
        assert len(_sig) == n_dim
        param_bounds = [(-s, s) for s in _sig]

    lhs_box = latin_hypercube(param_bounds, num_evaluations)
    T = _get_eigenbasis_transform(cov)
    return lhs_box.dot(T) + mu


def uniform_random_hypercube(param_bounds, num_evaluations, seed=None):
    """Generate a uniform random sampling oriented with the Cartesian axes.

    Parameters
    ----------
    param_bounds : n_dim-length sequence of 2-element tuples
        Each entry of param_bounds (min, max) specifies the bounds in each dimension

    num_evaluations : int
        Number of points in sample

    seed : int, optional
        Random number seed

    Returns
    -------
    sample : ndarray, shape(num_evaluations, n_dim)
        Uniform random sampling of a hypercube

    """

    param_bounds = np.atleast_2d(param_bounds)
    assert param_bounds.shape[1] == 2, shape_errmsg
    _dx = np.diff(param_bounds, axis=1)
    assert np.all(_dx > 0), minmax_errmsg
    num_params = param_bounds.shape[0]

    rng = np.random.RandomState(seed)
    unit_hypercube = rng.uniform(0, 1, num_params * num_evaluations)
    unit_hypercube = unit_hypercube.reshape((num_evaluations, num_params))

    xmins = param_bounds[:, 0]
    xmaxs = param_bounds[:, 1]

    params = np.zeros_like(unit_hypercube)
    for i in range(num_params):
        xmin, xmax = xmins[i], xmaxs[i]
        params[:, i] = xmin + (xmax - xmin) * unit_hypercube[:, i]
    return params
