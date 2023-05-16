from functools import partial
from jax.scipy.linalg import lu_factor

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
from jax.lax import cond, scan, dynamic_slice, custom_linear_solve
from jax.tree_util import Partial, tree_map
from jax.experimental.host_callback import id_tap
from tensorflow_probability.substrates.jax.distributions import WishartTriL

from util import tree_get_idx

# some lambdas
_identity = lambda x: x
tree_zeros_like = partial(tree_map, jnp.zeros_like)


def quad_form(x, A):
    return jnp.dot(x, jnp.matmul(A, x))


np.set_printoptions(linewidth=np.inf)
def array_print(arg, transforms):
    print(np.array2string(arg))


def jax_print(x):
    jd.print("{}", x)

def time_print(arg, transform):
    print(time.time())


def jax_time(x):
    id_tap(tap_func=time_print, arg=x)


def sample_wishart(key, v0, W0):
    W0_chol = jnp.linalg.cholesky(W0)
    return WishartTriL(v0, scale_tril=W0_chol).sample(seed=key)


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


def custom_trans_lu_solve(a, b, lu_factor):
    def _solve(matvec, x):
        return js.linalg.lu_solve(lu_factor, x)
    def _trans_solve(vecmat, x):
        return js.linalg.lu_solve(lu_factor, x, trans=1)
    matvec = partial(jnp.matmul, a.T)
    return custom_linear_solve(matvec, b, _trans_solve, _solve)


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


def custom_choL_solve(L, b, chol_factor):
    def _solve(matvec, x):
        return js.linalg.cho_solve((L, True), x)
    matvec = lambda _: L @ (L.T @ _)
    return custom_linear_solve(matvec, b, _solve, symmetric=True)


def comp_k_n(t1, t2, n1, n2, cov_fn, theta_cov):
    return cond(n1==n2, lambda a, b, c: cov_fn(a, b, c),
                lambda a, b, c: jnp.array(0.),
                t1, t2, tree_get_idx(theta_cov, n1))


@partial(jit, static_argnames=['cov_fn'])
def comp_K_N(t1, t2, cov_fn, theta_cov):
    N = theta_cov[0].shape[0]
    out = vmap(lambda a: vmap(
        lambda b: comp_k_n(t1, t2, a, b, cov_fn, theta_cov)
    )(jnp.arange(N)))(jnp.arange(N))
    return out


def K_N_diag(x, y, cov_fn, theta_cov, scaler):
    scaled_diag = cov_fn(x, y, theta_cov)/scaler
    return jnp.diag(scaled_diag)


def K_TN_blocks(x, y, cov_fn, theta_cov, scaler):
    return vmap(vmap(K_N_diag, in_axes=(None, 0, None, None, None)),
                in_axes=(0, None, None, None, None))(x, y, cov_fn, theta_cov,
                                                     scaler)


def K_N_diag_old(x, y, cov_fn, theta_cov, scaler):
    N = theta_cov[0].shape[0]
    scaled_diag = vmap(cov_fn, in_axes=(None, None, 0))(x, y, theta_cov)/scaler
    return jnp.diag(scaled_diag)


def K_TN_blocks_old(x, y, cov_fn, theta_cov, scaler):
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


def save_checkpoint(params, hist, train_args, is_inference=False):
    if not os.path.isdir(train_args.out_dir):
        os.mkdir(train_args.out_dir)
    relev_args_dict = {k: train_args.__dict__[k] for k
                       in train_args.__dict__ if k not in
                       ['out_dir', 'cv4a_dir', 'num_epochs_infer',
                        'eval_only', 'resume_ckpt', 'headless', 'plot_freq']}
    file_id = ["".join([k[0] for k in str(i).split('_')])+str(j)
           for i,j in zip(relev_args_dict.keys(),
                          relev_args_dict.values())]
    if is_inference:
        ckpt_file_name = "_".join(file_id) + "_inference_ckpt.pkl"
        hist_file_name = "_".join(file_id) + "_inference_hist.pkl"
    else:
        ckpt_file_name = "_".join(file_id) + "_ckpt.pkl"
        hist_file_name = "_".join(file_id) + "_hist.pkl"
    cloudpickle.dump(params, open(os.path.join(train_args.out_dir,
                                               ckpt_file_name), 'wb'))
    cloudpickle.dump(hist, open(os.path.join(train_args.out_dir,
                                             hist_file_name), 'wb'))


def load_checkpoint(train_args, is_inference=False):
    if not os.path.isdir(train_args.out_dir):
        os.mkdir(train_args.out_dir)
    relev_args_dict = {k: train_args.__dict__[k] for k in train_args.__dict__
                       if k not in ['out_dir', 'eval_only', 'resume_ckpt',
                                    'num_epochs_infer',
                                    'headless', 'plot_freq', 'cv4a_dir']}
    file_id = ["".join([k[0] for k in str(i).split('_')])+str(j)
           for i,j in zip(relev_args_dict.keys(),
                          relev_args_dict.values())]
    if is_inference:
        ckpt_file_name = "_".join(file_id) + "_inference_ckpt.pkl"
        hist_file_name = "_".join(file_id) + "_inference_hist.pkl"
    else:
        ckpt_file_name = "_".join(file_id) + "_ckpt.pkl"
        hist_file_name = "_".join(file_id) + "_hist.pkl"
    ckpt_file_path = os.path.join(train_args.out_dir, ckpt_file_name)
    hist_file_path = os.path.join(train_args.out_dir, hist_file_name)
    assert os.path.isfile(ckpt_file_path), "No checkpoint found for these settings!"
    assert os.path.isfile(hist_file_path), "No history found for these settings!"
    ckpt = pickle.load(open(ckpt_file_path, "rb"))
    hist = pickle.load(open(hist_file_path, "rb"))
    return ckpt, hist


if __name__=="__main__":
    print('nothing here!')
