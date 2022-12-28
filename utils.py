from functools import partial

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import jax.debug as jd

import numpy as np
import scipy as sp
import pdb
import time
import os
import cloudpickle
import pickle

from jax import vmap, jit
from jax.lax import cond, scan, dynamic_slice, custom_linear_solve, while_loop
from jax.tree_util import Partial
from jax.experimental.host_callback import id_tap
from jax._src.scipy.sparse.linalg import (
    _normalize_matvec,
    _sub,
    _vdot_real_tree
)

from util import tree_get_idx
from tensorflow_probability.substrates.jax.distributions import WishartTriL


def sample_wishart(key, v0, W0):
    W0_chol = jnp.linalg.cholesky(W0)
    return WishartTriL(v0, scale_tril=W0_chol).sample(seed=key)


def rdm_upper_cholesky(key, dim):
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
def jax_print(x):
    jd.print("{}", x)


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


def custom_solve(a, b, lu_factor):
    def _solve(matvec, x):
        return js.linalg.lu_solve(lu_factor, x)
    def _trans_solve(vecmat, x):
        return js.linalg.lu_solve(lu_factor, x, trans=1)
    matvec = partial(jnp.dot, a)
    return custom_linear_solve(matvec, b, _solve, _trans_solve)


def custom_triu_solve(u, b):
    def _solve(matvec, x):
        return js.linalg.solve_triangular(u, x)
    def _trans_solve(vecmat, x):
        return js.linalg.solve_triangular(u, x, trans=1)
    matvec = partial(jnp.dot, u)
    return custom_linear_solve(matvec, b, _solve, _trans_solve)


def custom_tril_solve(u, b):
    def _solve(matvec, x):
        return js.linalg.solve_triangular(u, x, lower=True)
    def _trans_solve(vecmat, x):
        return js.linalg.solve_triangular(u, x, trans=1, lower=True)
    matvec = partial(jnp.dot, u)
    return custom_linear_solve(matvec, b, _solve, _trans_solve)


def custom_chol_solve(a, b, chol_factor):
    def _solve(matvec, x):
        return js.linalg.cho_solve(chol_factor, x)
    matvec = partial(jnp.dot, a)
    return custom_linear_solve(matvec, b, _solve, symmetric=True)


def comp_k_n(t1, t2, n1, n2, cov_fn, theta_cov):
    return cond(n1==n2, lambda a, b, c: cov_fn(a, b, c),
                lambda a, b, c: jnp.array(0.),
                t1, t2, tree_get_idx(theta_cov, n1))


@partial(jit, static_argnames=['cov_fn'])
def comp_K_N(t1, t2, cov_fn, theta_cov):
    N = theta_cov[0].shape[0]
    out = jit(vmap(lambda a: vmap(
        lambda b: comp_k_n(t1, t2, a, b, cov_fn, theta_cov)
                         )(jnp.arange(N))))(jnp.arange(N))
    return out


@Partial(jit, static_argnames=['N', 'T'])
def get_diag_blocks(A, N, T):
    return vmap(lambda i: dynamic_slice(A, (i*N, i*N), (N, N)))(jnp.arange(T))


@Partial(jit, static_argnames=['N'])
def fill_triu(triu_elements, N):
    U = jnp.zeros((N, N))
    return U.at[jnp.triu_indices(N)].set(triu_elements)


@Partial(jit, static_argnames=['N'])
def fill_tril(tril_elements, N):
    L = jnp.zeros((N, N))
    return L.at[jnp.tril_indices(N)].set(tril_elements)


def matching_sources_corr(est_sources, true_sources, method="spearman"):
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


def save_checkpoint(params, hist, train_args):
    if not os.path.isdir(train_args.out_dir):
        os.mkdir(train_args.out_dir)
    relev_args_dict = {k: train_args.__dict__[k] for k
                       in train_args.__dict__ if k not in ['out_dir',
                        'eval_only', 'resume_ckpt', 'headless', 'plot_freq']}
    file_id = ["".join([k[0] for k in str(i).split('_')])+str(j)
           for i,j in zip(relev_args_dict.keys(),
                          relev_args_dict.values())]
    ckpt_file_name = "_".join(file_id) + "_ckpt.pkl"
    hist_file_name = "_".join(file_id) + "_hist.pkl"
    cloudpickle.dump(params, open(os.path.join(train_args.out_dir,
                                               ckpt_file_name), 'wb'))
    cloudpickle.dump(hist, open(os.path.join(train_args.out_dir,
                                             hist_file_name), 'wb'))


def load_checkpoint(train_args):
    if not os.path.isdir(train_args.out_dir):
        os.mkdir(train_args.out_dir)
    relev_args_dict = {k: train_args.__dict__[k] for k in train_args.__dict__
                       if k not in ['out_dir', 'eval_only', 'resume_ckpt',
                                    'headless', 'plot_freq']}
    file_id = ["".join([k[0] for k in str(i).split('_')])+str(j)
           for i,j in zip(relev_args_dict.keys(),
                          relev_args_dict.values())]
    ckpt_file_name = "_".join(file_id) + "_ckpt.pkl"
    ckpt_file_path = os.path.join(train_args.out_dir, ckpt_file_name)
    hist_file_name = "_".join(file_id) + "_hist.pkl"
    hist_file_path = os.path.join(train_args.out_dir, hist_file_name)
    assert os.path.isfile(ckpt_file_path), "No checkpoint found for these settings!"
    assert os.path.isfile(hist_file_path), "No history found for these settings!"
    ckpt = pickle.load(open(ckpt_file_path, "rb"))
    hist = pickle.load(open(hist_file_path, "rb"))
    return ckpt, hist


def mbcg_solve(A, B, x0=None, *, tol=1e-5, maxiter=None, M=None):
    def cond_fun(value, tol=tol, maxiter=maxiter):
        _, R, *_, j = value
        errs = jnp.sum(R**2, 0)**0.5
        return jnp.all(errs > tol) and (j < maxiter)


    def body_fun(value):
        U, triDs, R, D, Z, a, b, j = value
        _V = A(D)
        _a = (R*Z).sum(0) / (D*_V).sum(0)
        _U = U+jnp.diag(_a)@D
        _R = R-jnp.diag(_a)@_V
        _Z = M(_R)
        _b = (_Z*_Z).sum(0) / (Z*Z).sum(0)
        _D = _Z-jnp.diag(_b)@D
        _triDs = [triDs[i].at[j, j].set(
            1/_a[i] + cond(j == 0, lambda x, y: 0, lambda x, y: y/x, a[i], b[i])
        ) for i in range(len(triDs))]
        _triDs = [cond(j == 0, lambda x, y, z: z,
            lambda x, y, z: z.at[[j-1, j], [j, j-1]].set(jnp.sqrt(y)/x),
            _a[i], b[i], triDs[i]) for i in range(len(triDs))]
        return _U, _triDs, _R, _D, _Z, _a, _b, j+1


    n, t = B.shape
    U0 = jnp.zeros((n, t))
    R0 = _sub(A(U0), B)
    Z0 = M(R0)
    D0 = Z0.copy()
    triDs0 = [jnp.zeros((maxiter, maxiter)) for _ in range(t)]
    init_val = (U0, triDs0, R0, D0, Z0, jnp.zeros(t), jnp.zeros(t), 0)
    U_final, triDs_final, *_ = while_loop(cond_fun, body_fun, init_val)
    return U_final, triDs_final


@partial(jit, static_argnames=['max_rank'])
def pivoted_cholesky(A, tol, max_rank):
    def cond_fun(value, tol=tol, max_rank=max_rank):
        A, diag, perm, L, m = value
        perm_diag = diag[perm]
        error = jnp.sum(jnp.where(jnp.arange(perm_diag.shape[0]) >= m,
                                  perm_diag, 0))
        return (m < max_rank) & (error > tol)


    def body_fun(value):
        # this implementation is bit ugly -- jax doesnt allow dynamic indexing
        # so had to use masking and jnp.where
        A, diag, perm, L, m = value
        N = diag.shape[0]
        perm_diag = diag[perm]
        i = jnp.argmax(jnp.where(jnp.arange(N) >= m, perm_diag, -jnp.inf))
        perm_m = perm[m]
        perm_i = perm[i]
        perm = perm.at[m].set(perm_i).at[i].set(perm_m)
        max_val = jnp.sqrt(diag[perm[m]])
        L = L.at[perm[m], m].set(max_val)
        a_row = A[perm[m], :]
        L = L.at[perm, m].add(
            jnp.where(jnp.arange(N) >= m+1, a_row[perm], 0)
        )
        l_row = L[perm[m], :]
        l_row = jnp.where(jnp.arange(l_row.shape[0]) < m, l_row, 0)
        L_sub = l_row @ jnp.where((jnp.arange(N) >= m+1)[:, None],
                                  L[perm, :], 0).T
        L = L.at[perm, m].add(-L_sub)
        L = L.at[perm, m].set(jnp.where(jnp.arange(N)>=m+1, L[perm, m]/max_val,
                                        L[perm, m]))
        diag = diag.at[perm].set(jnp.where(jnp.arange(N) >= m+1,
                                           diag[perm]-L[perm, m]**2, diag))
        return (A, diag, perm, L, m+1)


    diag = jnp.diag(A)
    perm = jnp.arange(A.shape[0])
    pchol = jnp.zeros((A.shape[0], max_rank))
    init_val = (A, diag, perm, pchol, 0)
    *_, pchol, _ = while_loop(cond_fun, body_fun, init_val)
    return pchol


if __name__ == "__main__":
    key = jr.PRNGKey(8)
    Q = jr.normal(key, (8, 8))
    A = Q.T@Q

    pc = pivoted_cholesky(A, 1e-5, 8)
    pdb.set_trace()

