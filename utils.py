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

from jax import vmap, jit, device_put
from jax.lax import cond, scan, dynamic_slice, custom_linear_solve, while_loop
from jax.tree_util import Partial, tree_map
from jax.experimental.host_callback import id_tap
from jax._src.scipy.sparse.linalg import (
    _normalize_matvec,
    _sub,
    _vdot_real_tree
)

from util import tree_get_idx
from tensorflow_probability.substrates.jax.distributions import WishartTriL


# some lambdas
_identity = lambda x: x
tree_zeros_like = partial(tree_map, jnp.zeros_like)

def sample_wishart(key, v0, W0):
    W0_chol = jnp.linalg.cholesky(W0)
    return WishartTriL(v0, scale_tril=W0_chol).sample(seed=key)


def quad_form(x, A):
    return jnp.dot(x, jnp.matmul(A, x))


np.set_printoptions(linewidth=np.inf)
def jax_print(x):
    jd.print("{}", x)


def cho_invmp(x, y):
    return js.linalg.cho_solve(js.linalg.cho_factor(x), y)


def cho_inv(x):
    return cho_invmp(x, jnp.eye(x.shape[0]))


def lu_invmp(x, y):
    return js.linalg.lu_solve(js.linalg.lu_factor(x), y)


def lu_inv(x):
    return js.linalg.lu_solve(js.linalg.lu_factor(x), jnp.eye(x.shape[0]))


def custom_lu_solve(a, b, lu_factor):
    def _solve(matvec, x):
        return js.linalg.lu_solve(lu_factor, x)
    def _trans_solve(vecmat, x):
        return js.linalg.lu_solve(lu_factor, x, trans=1)
    matvec = partial(jnp.matmul, a)
    return custom_linear_solve(matvec, b, _solve, _trans_solve)


def custom_triu_solve(u, b):
    def _solve(matvec, x):
        return js.linalg.solve_triangular(u, x)
    def _trans_solve(vecmat, x):
        return js.linalg.solve_triangular(u, x, trans=1)
    matvec = partial(jnp.matmul, u)
    return custom_linear_solve(matvec, b, _solve, _trans_solve)


def custom_tril_solve(u, b):
    def _solve(matvec, x):
        return js.linalg.solve_triangular(u, x, lower=True)
    def _trans_solve(vecmat, x):
        return js.linalg.solve_triangular(u, x, trans=1, lower=True)
    matvec = partial(jnp.dot, u)
    return custom_linear_solve(matvec, b, _solve, _trans_solve)


def custom_cho_solve(a, b, cho_factor):
    def _solve(matvec, x):
        return js.linalg.cho_solve(cho_factor, x)
    matvec = partial(jnp.matmul, a)
    return custom_linear_solve(matvec, b, _solve, symmetric=True)


def comp_k_n(t1, t2, n1, n2, cov_fn, theta_cov):
    return cond(n1==n2, lambda a, b, c: cov_fn(a, b, c),
                lambda a, b, c: jnp.array(0.), t1, t2, tree_get_idx(theta_cov, n1))


@partial(jit, static_argnames=['cov_fn'])
def comp_K_N(t1, t2, cov_fn, theta_cov):
    N = theta_cov[0].shape[0]
    out = vmap(lambda a: vmap(
        lambda b: comp_k_n(t1, t2, a, b, cov_fn, theta_cov)
    )(jnp.arange(N)))(jnp.arange(N))
    return out


def K_N_diag(x, y, cov_fn, theta_cov, scaler):
    N = theta_cov[0].shape[0]
    scaled_diag = vmap(cov_fn, in_axes=(None, None, 0))(x, y, theta_cov)/scaler
    return jnp.diag(scaled_diag)


def K_TN_blocks(x, y, cov_fn, theta_cov, scaler):
    return vmap(vmap(K_N_diag, in_axes=(None, 0, None, None, None)),
                in_axes=(0, None, None, None, None))(x, y, cov_fn, theta_cov,
                                                     scaler)


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


#@partial(jit, static_argnames=['A', 'M', 'maxiter'])
def _mbcg_solve(A, B, x0=None, *, tol=0.01, maxiter=None, M=None):
    def cond_fun(value):
        *_, R, j = value
        errs = jnp.sum(R**2, 0)**0.5
        jax_print((j, errs))
        return jnp.any(errs > tol) & (j < maxiter)


    def body_fun(value):
        U, D, Z, a, b, R, j = value
        _V = A(D)
        _a = (R*Z).sum(0) / (D*_V).sum(0)
        _U = U + _a.reshape(1, -1)*D
        _R = R - _a.reshape(1, -1)*_V
        _Z = M(_R)
        _b = (_R*_Z).sum(0) / (R*Z).sum(0)
        _D = _Z + _b.reshape(1, -1)*D
        return _U, _D, _Z, _a, _b, _R, j+1


    B = B.reshape(B.shape[0], -1)
    n, t = B.shape
    R0 = _sub(B, A(x0))
    Z0 = M(R0)
    D0 = Z0.copy()
    init_val = (x0, D0, Z0, jnp.zeros(t), jnp.zeros(t), R0, 0)
    U_final, *_ = while_loop(cond_fun, body_fun, init_val)
    return U_final


def mbcg(A, B, x0=None, *, tol=0.01, maxiter=None, M=None):
    # modifying https://jax.readthedocs.io/en/latest/_modules/jax/_src/\
    #scipy/sparse/linalg.html#cg
    # Note: assume A is symmetric
    if x0 is None:
        x0 = tree_map(jnp.zeros_like, B)

    B, x0 = device_put((B, x0))

    if maxiter is None:
        size = sum(bi.size for bi in tree_leaves(B))
        maxiter = 10*size

    if M is None:
        M = _identity

    mbcg_solve = partial(_mbcg_solve, x0=x0, tol=tol, maxiter=maxiter, M=M)
    x = custom_linear_solve(A, B, solve=mbcg_solve, symmetric=True,
                            has_aux=False)
    return x#, tri_diags


def pivoted_cholesky(A, max_rank, tol=0.0):
    # this implementation is bit ugly -- jax doesnt allow dynamic indexing
    # so had to use masking and jnp.where
    # also had to set tol=0.0 as otherwise reach issues wiht dynamic slicign
    def cond_fun(value, max_rank=max_rank, tol=tol):
        diag, perm, L, m = value
        del L
        perm_diag = diag[perm]
        error = jnp.sum(jnp.where(jnp.arange(perm_diag.shape[0]) >= m,
                                  perm_diag, 0))
        return (m < max_rank) & (error > tol)


    def body_fun(value):
        diag, perm, L, m = value
        N = diag.shape[0]
        perm_diag = diag[perm]
        i = jnp.argmax(jnp.where(jnp.arange(N) >= m, perm_diag, -jnp.inf))
        perm_m = perm[m]
        perm_i = perm[i]
        perm = perm.at[m].set(perm_i).at[i].set(perm_m)
        max_val = jnp.sqrt(diag[perm[m]])
        l = jnp.zeros((N,)).at[perm[m]].set(max_val)
        l = l.at[perm].add(jnp.where(jnp.arange(N) >= m+1, A[perm[m]][perm],0))
        l_row = jnp.where(jnp.arange(L.shape[1]) < m, L[perm[m], :], 0)
        l_sub = l_row @ jnp.where((jnp.arange(N) >= m+1)[:, None],
                                  L[perm, :], 0).T
        l = l.at[perm].add(-l_sub)
        l = l.at[perm].set(jnp.where(jnp.arange(N)>=m+1, l[perm]/max_val,
                                        l[perm]))
        diag = diag.at[perm].set(jnp.where(jnp.arange(N) >= m+1,
                                           diag[perm]-l[perm]**2, diag))
        L = L.at[:, m].set(l)
        return (diag, perm, L, m+1)


    diag = jnp.diag(A)
    perm = jnp.arange(A.shape[0])
    pchol = jnp.zeros((A.shape[0], max_rank))
    init_val = (diag, perm, pchol, 0)
    *_, pchol, m = while_loop(cond_fun, body_fun, init_val)
    return pchol


def solve_precond_plus_block_diag(L, D, B):
    # D must be a batch of block diagonal matrices
    DB = vmap(jnp.matmul)(D, B.reshape(D.shape[0], -1, B.shape[-1])).reshape(
        -1, B.shape[-1])
    DL = vmap(jnp.matmul)(D, L.reshape(D.shape[0], -1, L.shape[-1])).reshape(
        -1, L.shape[-1])
    A = jnp.eye(L.shape[1])+L.T@DL
    cho_factor = js.linalg.cho_factor(A)
    woodbury_inv = custom_cho_solve(A, L.T@DB, cho_factor)
    return DB - DL@woodbury_inv


def solve_precond_plus_diag(L, d, B):
    # B must be a batch of block diagonal matrices
    dB = d[:, None]*B
    dL = d[:, None]*L
    A = jnp.eye(L.shape[1])+L.T@dL
    cho_factor = js.linalg.cho_factor(A)
    woodbury_inv = custom_cho_solve(A, L.T@dB, cho_factor)
    return dB - dL@woodbury_inv


if __name__ == "__main__":
    key = jr.PRNGKey(8)
    Q = jr.normal(key, (8, 8))
    A = Q.T@Q
    B = jr.normal(jr.split(key)[1], (8, 3))

    pc = pivoted_cholesky(A, 1e-9, 8)

    Pinv_fun = lambda _: solve_precond_plus_diag(pc, 0.01*jnp.ones(pc.shape[0]), _)
    A_fun = partial(jnp.matmul, A)
    out = mbcg(A_fun, B, tol=1e-30, maxiter=53, M=Pinv_fun)

    jax_print(jnp.abs(jnp.linalg.inv(A)@B - out).sum())

    pdb.set_trace()

