from functools import partial
from jax.config import config
#config.update("jax_enable_x64", True)


import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import jax.debug as jdb
import optax

import numpy as np
import scipy as sp
import pdb
import time
import os
import cloudpickle
import pickle

from jax import vmap, jit, device_put, grad, lax, block_until_ready
from jax.lax import (
    cond,
    custom_linear_solve,
    while_loop,
    top_k,
    scan,
    fori_loop
)
from jax.tree_util import Partial, tree_map, tree_leaves
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
eye_like = lambda x: jnp.eye(x.shape[0])

def sample_wishart(key, v0, W0):
    W0_chol = jnp.linalg.cholesky(W0)
    return WishartTriL(v0, scale_tril=W0_chol).sample(seed=key)


def quad_form(x, A):
    return jnp.dot(x, jnp.matmul(A, x))


np.set_printoptions(linewidth=np.inf)
def jax_print(x):
    jdb.print("{}", x)


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


def custom_choL_solve(L, b):
    def _solve(matvec, x):
        return js.linalg.cho_solve((L, True), x)
    matvec = lambda _: L @ (L.T @ _)
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
    scaled_diag = cov_fn(x, y, theta_cov)/scaler
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


# reorth
def _cg_reorth(r, R, do_ro):
    return cond(do_ro, lambda _: ((r*_).sum(0) / (_**2).sum(0))[None, :] * _,
                lambda _: jnp.zeros_like(_), R)


def _mbcg_solve(A, B, x0=None, *, tol=0.01, maxiter=None, M=None):
    def cond_fun(value):
        *_, R, j = value
        errs = jnp.sum(R**2, 0)**0.5
        #jax_print((j, jnp.max(errs)))
        return (j < maxiter) #& jnp.any(errs > tol)


    def body_fun(value):
        X, a_all, b_all, P, Z, R, j = value
        _V = A(P)
        _a = (R*Z).sum(0) / (P*_V).sum(0)
        _X = X + _a.reshape(1, -1)*P
        _R = R - _a.reshape(1, -1)*_V
        _Z = M(_R)
        _b = (_R*_Z).sum(0) / (R*Z).sum(0)
        _P = _Z + _b.reshape(1, -1)*P
        _a_all = a_all.at[j].set(_a)
        _b_all = b_all.at[j].set(_b)
        return _X, _a_all, _b_all, _P, _Z, _R, j+1


    n, t = B.shape
    R0 = _sub(B, A(x0))
    Z0 = M(R0)
    P0 = Z0.copy()
    a_all = jnp.zeros((maxiter, t))
    b_all = jnp.zeros((maxiter, t))
    init_val = (x0, a_all, b_all, P0, Z0, R0, 0)
    X, alphas, betas, *_ = while_loop(cond_fun, body_fun, init_val)
    b_a = jnp.sqrt(betas[:-1])/alphas[:-1]
    Ts_off = vmap(lambda _: jnp.diag(_, k=1), in_axes=(1,))(b_a)
    Ts = vmap(jnp.diag, in_axes=(1,))(1/alphas +
        jnp.vstack((jnp.zeros((1, t)), betas[:-1]/alphas[:-1])))
    Ts = Ts + Ts_off + Ts_off.swapaxes(1, 2)
    #jax_print((jnp.ones(2), jnp.linalg.eigh(Ts)[0]))
    return X, Ts


def mbcg(A, B, x0=None, *, tol=0.0, maxiter=None, M=None):
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
    x, Ts = custom_linear_solve(A, B, solve=mbcg_solve, symmetric=True,
                                has_aux=True)
    return x, Ts


def solve_precond_plus_diag(L, d, B):
    # B must be a batch of block diagonal matrices
    dB = d[:, None]*B
    dL = d[:, None]*L
    A = jnp.eye(L.shape[1])+L.T@dL
    cho_factor = js.linalg.cho_factor(A)
    woodbury_inv = custom_cho_solve(A, L.T@dB, cho_factor)
    return dB - dL@woodbury_inv


@partial(jit, static_argnames=['k'])
def naive_top_k(data, k):
    """Top k implementation built with argmax. Since lax.top_k is slow.
    Adapted from: https://github.com/google/jax/issues/9940
    Faster for smaller k."""

    def top_1(data):
        idx = jnp.argmax(data)
        value = data[idx]
        data = data.at[idx].set(-jnp.inf)
        return data, value, idx

    def scannable_top_1(carry, unused):
        data = carry
        data, value, indice = top_1(data)
        return data, (value, indice)

    data, (values, indices) = scan(scannable_top_1, data, (), k)
    return values.T, indices.T


#@partial(jit, static_argnames=['nz_max', 'Minv_f'])
def fsai(A, num_iter, nz_max, eps, G0, Minv_f):
    n = A.shape[1]

    def _G_i_update_fun(k, value):
        i, g_i, phi_grad, p, idx = value
        idx = naive_top_k(jnp.abs(phi_grad), nz_max)[1]
        r = A[idx].T @ p[idx]
        alpha = jnp.dot(p[idx], phi_grad[idx])/jnp.dot(p[idx], r[idx])
        alpha = cond(jnp.isnan(alpha), tree_zeros_like, _identity, alpha)
        alpha = cond(jnp.isinf(alpha), tree_zeros_like, _identity, alpha)
        _g_i = g_i + alpha*p
        idx = naive_top_k(jnp.abs(_g_i), nz_max)[1]
        g_i_new = jnp.zeros_like(_g_i).at[idx].set(_g_i[idx])
        # below done just in case diag falls out of top_k (?shouldnt happen?) 
        g_i_new = g_i_new.at[i].set(_g_i[i])
        #phi_grad = jnp.where(jnp.arange(n) < i, 2*A.T @ g_i_new, 0)
        phi_grad = jnp.where(jnp.arange(n) < i,
                 jnp.zeros_like(g_i).at[idx].set((phi_grad-alpha*r)[idx]), 0)
        p = Minv_f(phi_grad)
        return (i, g_i_new, phi_grad, p, idx)


    def _calc_G_i(i, G0_i):
        idx = naive_top_k(jnp.abs(G0_i), nz_max)[1]
        grad = -A[idx].T @ G0_i[idx]
        phi_grad = jnp.where(jnp.arange(n) < i, grad, 0)
        p = Minv_f(phi_grad)
        init_val = (i, G0_i, phi_grad, p, idx)
        i, Gk_i, *_, idx = fori_loop(0, num_iter, _G_i_update_fun, init_val)
        d_ii = quad_form(Gk_i[idx], A[idx][:, idx])**-0.5
        return d_ii*Gk_i, idx


    if G0 == None:
        G0_tilde = jnp.eye(A.shape[0])
    else:
        G0_tilde = (1/jnp.einsum('ii->i', G0))[:, None] * G0

    G, idx = vmap(_calc_G_i)(jnp.arange(A.shape[0]), G0_tilde)
    return G, idx


def lanczos_tridiag(A, v1, m):
    '''Performs Lanczos tridiagonalization of pos.def. matrix.'''
    def _lanczos_step(carry, x):
        v0, v1, b1 = carry
        v = A(v1) - b1*v0
        a1 = jnp.dot(v1, v)
        v = v - a1*v1
        b2 = jnp.linalg.norm(v)
        v2 = v / b2
        return (v1, v2, b2), (v1, a1, b1)

    d = v1.shape[0]
    v0 = jnp.zeros((d,))
    b1 = 0.
    _, (V, alphas, betas) = scan(_lanczos_step, (v0, v1, b1), None, length=m)
    T_off = jnp.diag(betas[1:], k=1)
    T = jnp.diag(alphas)+T_off+T_off.T
    return T, V.T


def _ro(r, q_k, do_ro):
    return cond(do_ro, lambda _: _*jnp.dot(r, _),
                lambda _: jnp.zeros_like(_), q_k)


def reorth(r, Q, w, macheps):
    ro_idx = w > macheps**(3/4)  # hacky -- fix!
    ro = vmap(_ro, (None, 0, 0))(r, Q, ro_idx)
    r = r - ro.sum(0)
    return r, ro_idx


def lanczos_pro(key, A, r1, m, macheps=2**-52):
    '''Performs Lanczos tridiagonalization of pos.def. matrix.'''
    def _lanczos_step(carry, i):
        key, r1, b1, q0, all_a, all_b, all_Q, W, y = carry
        all_b = all_b.at[i].set(b1)
        q1 = r1 / b1
        all_Q = all_Q.at[i].set(q1)
        u1 = A(q1) - b1*q0
        a1 = jnp.dot(u1, q1)
        all_a = all_a.at[i].set(a1)
        r2 = u1 - a1*q1
        _b2 = jnp.linalg.norm(r2)

        # update W
        key, key_a, key_b, key_c = jr.split(key, 4)
        bw1 = all_b[1:-1]*W[-1, 1:-1]
        aw = (all_a[:-2] - all_a[i])*W[-1, :-2]
        bw2 = all_b[:-2]*jnp.concatenate((jnp.zeros((1,)), W[-1, :-3]))
        bw3 = all_b[i]*W[-2,:-2]
        var_theta = macheps*(all_b[1:-1]+_b2)*(0.3**0.5)*jr.normal(key_a, (1,))
        w_new = jnp.where(jnp.arange(m-2) < i, (bw1+aw+bw2-bw3)/_b2
                          + var_theta, 0)
        W = W.at[-2].set(W[-1])
        W = W.at[-1, :-2].set(w_new)
        scale = cond(i == 0, lambda _: 1., lambda _: _[1]/_b2, all_b)
        W = W.at[-1, i].set(macheps*d*scale*(0.6**0.5)*jr.normal(key_b))
        W = W.at[-1, i+1].set(1.)

        # pro
        w_ref = jnp.where(jnp.arange(m) < i, jnp.abs(W[-1]), 0)
        do_pro = (w_ref.max() > jnp.sqrt(macheps)) | (y == 1)
        r2, ro_idx = cond(do_pro, reorth, lambda *_: (r2, jnp.array([False]*(m))),
                          lax.stop_gradient(r2), lax.stop_gradient(all_Q),
                          lax.stop_gradient(w_ref), macheps)
        b2 = jnp.linalg.norm(r2)
        W = W.at[-1].set(jnp.where(
            ro_idx, macheps*(1.5**0.5)*jr.normal(key_c, (m,)), W[-1]))
        y = (do_pro & (y == 0))*1
        return (key, r2, _b2, q1, all_a, all_b, all_Q, W, y), (q1, a1, b1)


    d = r1.shape[0]
    q0 = jnp.zeros((d,))
    b1 = jnp.linalg.norm(r1)
    all_a = jnp.zeros((m,))
    all_b = jnp.zeros((m,))
    all_Q = jnp.zeros((m, d))
    y = 0
    W = jnp.zeros((2, m))
    W = W.at[-1, 0].set(1.)
    *_ , (V, alphas, betas) = scan(_lanczos_step,
                                   (key, r1, b1, q0, all_a, all_b,
                                    all_Q, W, y),
                                   jnp.arange(m))
    T_off = jnp.diag(betas[1:], k=1)
    T = jnp.diag(alphas)+T_off+T_off.T
    return T, V.T


def krylov_subspace_sampling(key, A, v1, m):
    ''' Samples N(0, inv(A))'''
    #import numpy as np
    b = jnp.linalg.norm(v1)
    v1 = v1 / b
    T, V = lanczos_pro(key, A, v1, m)
    # use svd here as svd=eigh for p.d matrices, but ensures >0 sing vals

    #eV, ew, _ = jnp.linalg.svd(T, full_matrices=False)
    ew, eV = jnp.linalg.eigh(T)

    T_neg_sqrt_v1 = eV @ ((ew**-0.5)*eV[0])
    return b*(V @ T_neg_sqrt_v1)


if __name__ == "__main__":
    key = jr.PRNGKey(8)
    Q = jr.normal(key, (6000, 6000))
    A = Q.T@Q
    print(jnp.linalg.eigh(A)[0])
    key1, key0 = jr.split(key)
    v1 = jr.normal(key0, (A.shape[0],))
    T, _ = lanczos_pro(jr.split(key)[1], lambda _: A@_, v1, 100, macheps=2**-52)

