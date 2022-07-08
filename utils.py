from jax.scipy.linalg import lu_factor
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js

import numpy as np
import scipy as sp
import pdb
import time

from jax import vmap, jit
from jax.lax import cond
from jax.experimental.host_callback import id_tap
from util import tree_get_idx


def rdm_upper_cholesky_of_precision(key, dim):
    P = jr.orthogonal(key, dim)
    key, _ = jr.split(key)
    Q = jnp.diag(1/jr.uniform(key, shape=(dim,)))
    precision_mat = jnp.dot(P.T, jnp.dot(Q, P))
    L = jnp.linalg.cholesky(precision_mat)
    return L.T


def reorder_covmat(cov, N, square=True):
    T_l, T_r = tuple(jnp.int64(_/N) for _ in cov.shape)
    P = jnp.zeros((cov.shape[0], cov.shape[0]))
    for t in range(T_l):
        for n in range(N):
            P = P.at[t*N+n, n*T_l+t].set(1.)
    if square:
        P2 = P
    elif not square:
        P2 = jnp.zeros((cov.shape[1], cov.shape[1]))
        for t in range(T_r):
            for n in range(N):
                P2 = P2.at[t*N+n, n*T_r+t].set(1.)
    return jnp.dot(P, jnp.dot(cov, P2.T))


np.set_printoptions(linewidth=np.inf)
def array_print(arg, transforms):
    print(np.array2string(arg))


def jax_print(x):
    id_tap(tap_func=array_print, arg=x)


def time_print(arg, transform):
    print(time.time())


def jax_time(x):
    id_tap(tap_func=time_print, arg=x)


def cho_invmp(x, y):
    return js.linalg.cho_solve(js.linalg.cho_factor(x), y)


def cho_inv(x):
    return cho_invmp(x, jnp.eye(x.shape[0]))


def lu_invmp(x, y):
    return js.linalg.lu_solve(js.linalg.lu_factor(x), y)


def lu_inv(x):
    return js.linalg.lu_solve(js.linalg.lu_factor(x), jnp.eye(x.shape[0]))


def comp_k_n(t1, t2, n1, n2, cov_fn, theta_cov):
    return jit(cond)(n1==n2, lambda a, b, c: cov_fn(a, b, c),
                lambda a, b, c: jnp.array(0.),
                t1, t2, tree_get_idx(theta_cov, n1))


def comp_K_N(t1, t2, cov_fn, theta_cov):
    N = theta_cov[0].shape[0]
    out = jit(vmap(lambda a: vmap(
        lambda b: comp_k_n(t1, t2, a, b, cov_fn, theta_cov)
                         )(jnp.arange(N))))(jnp.arange(N))
    return out


def matching_sources_corr(est_sources, true_sources, method="pearson"):
    """Finding matching indices between true and estimated sources.
    Args:
        est_sources (array): data on estimated independent components.
        true_sources (array): data on true independent components.
        method (str): "pearson" or "spearman" correlation method to use.
    Returns:
        mean_abs_corr (array): average correlation matrix between
                               matched sources.
        s_est_sort (array): estimed sources array but columns sorted
                            according to best matching index.
        cid (array): vector of the best matching indices.
    """
    N = est_sources.shape[0]

    # calculate correlations
    if method == "pearson":
        corr = np.corrcoef(true_sources, est_sources, rowvar=True)
        corr = corr[0:N, N:]
    elif method == "spearman":
        corr, _ = sp.stats.spearmanr(true_sources, est_sources, axis=1)
        corr = corr[0:N, N:]

    # sort variables to try find matching components
    ridx, cidx = sp.optimize.linear_sum_assignment(-np.abs(corr))

    # calc with best matching components
    mean_abs_corr = np.mean(np.abs(corr[ridx, cidx]))
    s_est_sorted = est_sources[cidx, :]
    return mean_abs_corr, s_est_sorted, cidx


def plot_ic(s_n, s_est_n, ax, ax1):
    T = s_n.shape[0]
    ax.clear()
    #ax1.clear()
    ax.plot(s_n, color='blue')
    ax.set_xlim([0, T])
    #ax.set_ylim([-2, 2])
    #ax1.plot(s_est_n, color='red')


if __name__=="__main__":
    pdb.set_trace()
