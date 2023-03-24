from jax._src.scipy.linalg import block_diag
from jax.numpy.linalg import slogdet

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import jax.debug as jdb
import pdb

from jax import vmap, jit, lax, device_put, block_until_ready
from jax.tree_util import tree_map, Partial

from kernels import (
    bound_se_kernel_params,
    squared_euclid_dist_mat
)
from utils import (
    custom_lu_solve,
    custom_tril_solve,
    custom_choL_solve,
    comp_K_N,
    K_N_diag,
    K_TN_blocks,
    quad_form,
    fill_tril,
    jax_print,
    pivoted_cholesky,
    make_pinv_block_cho_version,
    solve_precond_plus_diag,
    mbcg,
    _identity,
    fsai,
    fsai2,
    lanczos_tridiag
)
from util import *
from gamma import *
from gaussian import *

from functools import partial
from time import perf_counter


def structured_elbo_s(key, theta, phi_s, logpx, cov_fn, x, t, tau, nsamples,
                      K, G):
    n_s_samples, _, max_precond_rank, max_cg_iters, n_probe_vecs = nsamples
    theta_x, _ = theta[:2]
    L, h = phi_s
    N = h.shape[1]
    T = t.shape[0]

    # scale covariance
    K = K / tau[None, None, None, :]

    #1: compute parameters for \tilde{q(s|tau)}
    L = vmap(fill_tril, in_axes=(0, None))(L, N)
    L_inv = vmap(custom_tril_solve, (0, None))(L, jnp.eye(N))
    J_inv = jnp.matmul(L_inv.swapaxes(1, 2), L_inv)
    A = K.at[jnp.arange(T), jnp.arange(T)].add(J_inv).swapaxes(
      1, 2).reshape(N*T, N*T)
    K = K.swapaxes(1, 2).reshape(N*T, N*T)

    # scale preconditioner of inverse(cho_factor(K)) with current tau samples
    G = jnp.tile(jnp.sqrt(tau), T)[:, None] * G


    def A_mvp(x):
        if len(x.shape) == 1:
            Jinvx = vmap(custom_choL_solve)(
                L, x.reshape(L.shape[0], -1)).reshape(-1)
        elif len(x.shape) == 2:
            Jinvx = vmap(custom_choL_solve)(
                L, x.reshape(L.shape[0], -1, x.shape[-1])).reshape(
                    -1, x.shape[-1])
        return Jinvx + K@x


    # calculate preconditioner for inverse(cho_factor(A))
    P = fsai(A, 5, 10, 1e-8, None, _identity)
    Minv_mvp = lambda b: jnp.matmul(P.T, jnp.matmul(P, b))

    ## set up an run mbcg
    B = K@h.reshape(-1, 1)
    solves, T_mats = mbcg(A_mvp, B, maxiter=max_cg_iters, M=Minv_mvp)

    # compute vlb terms
    m = vmap(custom_choL_solve)(
        L, solves2[:, 0].reshape(L.shape[0], -1)
    ).reshape(-1)


    # checking with exact
    J = jnp.matmul(L, L.swapaxes(1, 2))
    m = jnp.linalg.solve(js.linalg.block_diag(*J) + jnp.linalg.inv(K),
                         h.reshape(-1))


    import scipy as sp
    import numpy as np
    jdb.breakpoint()
    #pdb.set_trace()

    # sample probe vectors with preconditioner covariance 
#    key, zk_key, zl_key = jr.split(key, 3)
#    z_K = P_K_lower @ jr.normal(zk_key, (P_K_lower.shape[1], n_probe_vecs))
#    z_Linv = W_inv.swapaxes(1, 2) @ jr.normal(zl_key, (T, W_inv.shape[1],
#                                                       n_probe_vecs))
#    z = z_K + z_Linv.reshape(-1, n_probe_vecs)
#    z = z / jnp.linalg.norm(z, 2, axis=0, keepdims=True)
#

#
#    # compute quadratic form term of normalizer
#    JinvKm = solves[:, 0]
#    Linvm = vmap(jnp.matmul)(Linv, m).reshape(-1)
#    quad_term = 0.5 * Linvm@JinvKm
#
#    ## compute logdet terms of normalizer
#    ew, eV = jnp.linalg.eigh(Ts)
#    E_tr_delta_log = N*T * jnp.sum(jnp.log(ew) * eV[:, 0, :]**2, 1).mean()
#    logdet_P_K = jnp.linalg.slogdet(LtWWtL+jnp.eye(LtWWtL.shape[0]))[1]
#    logdet = -0.5*logdet_P_K-0.5*E_tr_delta_log

    #A_inv = jnp.linalg.inv(jnp.linalg.inv(K)+J)
#    logZ = 0.5*h.T@A_inv@h + 0.5*jnp.linalg.slogdet(A_inv)[1] - \
#        0.5*jnp.linalg.slogdet(K)[1]

    # set preconditioners and func to calculate its inverse matrix product
    #    #
    # run mbcg
    #A_fun = partial(jnp.matmul, K_Jinv)
    #logZ2 = 0.5*(h.T@Jinv)@jnp.linalg.inv(Jinv+K)@(K@h) - 0.5*jnp.linalg.slogdet(
    #    Jinv+K)[1] + 0.5*jnp.linalg.slogdet(Jinv)[1]

#    jax_print(logZ)
    #jax_print(logZ2)


    #WTy = jnp.einsum('ijk,ik->jk', What, yhat).T.reshape(-1, 1)
    #L = js.linalg.block_diag(*jnp.moveaxis(
    #  jnp.einsum('ijk, ilk->jlk', What, What), -1, 0))
    #LK = L@Kuu
    #lu_fact = jit(js.linalg.lu_factor)(jnp.eye(L.shape[0])+LK)
    #KyyWTy = custom_solve(jnp.eye(L.shape[0])+LK, WTy, lu_fact)
    #mu_s = Ksu @ KyyWTy
    #cov_solve = custom_solve(jnp.eye(L.shape[0])+LK, L, lu_fact)
    #cov_s = vmap(lambda X, y: jnp.diag(y)-X@cov_solve@X.T,
    #      in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss)
    #s, rng = rngcall(lambda _: jr.multivariate_normal(_, mu_s.reshape(T, N),
    #    cov_s, shape=(nsamples, T)), rng)

    ## compute E_{\tilde{q(s|tau)}}[log_p(x_t|s_t)]
    #Elogpx = jnp.mean(
    #    jnp.sum(vmap(lambda _: vmap(logpx, (1, 0, None))(x, _, theta_x))(s), 1)
    #)

    ## compute KL[q(u)|p(u)]
    #tr = jnp.trace(js.linalg.lu_solve(lu_fact, LK.T, trans=1).T)
    #h = Kuu@KyyWTy
    #logZ = 0.5*(jnp.dot(WTy.squeeze(), h)
    #            -jnp.linalg.slogdet(jnp.eye(L.shape[0])+LK)[1])
    #KLqpu = -0.5*(tr+h.T@L@h)+WTy.T@h - logZ
    s, _ = rngcall(lambda _: jr.multivariate_normal(_, jnp.zeros((T, N)),
                jnp.eye(N), shape=(n_s_samples, T)), key)
    return jnp.zeros((0,)), s +lax.stop_gradient(G).sum()+lax.stop_gradient(P).sum()
                      #Elogpx-KLqpu, s


# compute elbo estimate, assumes q(tau) is gamma
def structured_elbo(rng, theta, phi, logpx, cov_fn, x, t, nsamples, K, G):
    _, nsamples_tau, *_ = nsamples
    theta_tau = theta[2]
    theta_tau = 2.+jnp.exp(theta_tau)
    phi_s, phi_tau = phi[:2]
    N = phi_tau[0].shape[0]
    # in case df param is replicated to be same for all ICs
    theta_tau = theta_tau.repeat(N-theta_tau.shape[0]+1)
    # to avoid numerical issues
    phi_tau = tree_map(jnp.exp, phi_tau)
    tau, rng = rngcall(gamma_sample, rng, gamma_natparams_fromstandard(phi_tau),
                       (nsamples_tau, *phi_tau[0].shape))
    kl = jnp.sum(
        gamma_kl(
            gamma_natparams_fromstandard(phi_tau),
            gamma_natparams_fromstandard((theta_tau/2, theta_tau/2))), 0
    )
    vlb_s, s = vmap(structured_elbo_s, (None, None, None, None, None, None,
             None, 0, None, None, None))(rng, theta, phi_s, logpx, cov_fn,
                                         x, t, tau, nsamples, K, G)
    return jnp.mean(vlb_s, 0) - kl, s


def gp_elbo(rng, theta, phi_s, logpx, cov_fn, x, t, nsamples):
    theta_x, theta_cov = theta[:2]
    What, yhat, tu = phi_s
    N, n_pseudo = yhat.shape
    T = t.shape[0]
    theta_cov = tree_map(lambda _: jnp.exp(_), theta_cov)
    # repeat in case the same kernel replicated for all ICs
    theta_cov = tree_map(lambda _: _.repeat(N-theta_cov[0].shape[0]+1),
                         theta_cov)
    t_dist_mat = jnp.sqrt(squared_euclid_dist_mat(t))
    theta_cov = bound_se_kernel_params(
        theta_cov, sigma_min=1e-3,
        ls_min=jnp.min(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)]),
        ls_max=jnp.max(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)])
    )
    Kuu = vmap(lambda b: vmap(lambda a:
        comp_K_N(a, b, cov_fn, theta_cov)
    )(tu))(tu)
    Kuu = Kuu.swapaxes(1, 2).reshape(n_pseudo*N, n_pseudo*N)
    Ksu = vmap(lambda b:vmap(lambda a:
        comp_K_N(a, b, cov_fn, theta_cov))(tu))(t)
    Ksu = Ksu.swapaxes(1, 2).reshape(T*N, n_pseudo*N)
    kss = vmap(
        lambda tc: vmap(lambda t: cov_fn(t, t, tc))(t)
        )(theta_cov)

    # compute parameters for \tilde{q(s)}
    What = vmap(fill_triu, in_axes=(1, None), out_axes=-1)(What, N)
    WTy = jnp.einsum('ijk,ik->jk', What, yhat).T.reshape(-1, 1)
    L = js.linalg.block_diag(*jnp.moveaxis(
      jnp.einsum('ijk, ilk->jlk', What, What), -1, 0))
    LK = L@Kuu
    lu_fact = jit(js.linalg.lu_factor)(jnp.eye(L.shape[0])+LK)
    KyyWTy = custom_solve(jnp.eye(L.shape[0])+LK, WTy, lu_fact)
    #KyyWTy = js.linalg.lu_solve(lu_fact, WTy)
    mu_s = Ksu @ KyyWTy
    cov_solve = custom_solve(jnp.eye(L.shape[0])+LK, L, lu_fact)
    cov_s = vmap(lambda X, y: jnp.diag(y)-X@cov_solve@X.T,
          in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss)
    #cov_s = vmap(lambda X, y: jnp.diag(y)-X@js.linalg.lu_solve(lu_fact, L)@X.T,
    #      in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss)
    s, rng = rngcall(lambda _: jr.multivariate_normal(_, mu_s.reshape(T, N),
        cov_s, shape=(nsamples, T)), rng)

    # compute E_{\tilde{q(s)}}[log_p(x_t|s_t)]
    Elogpx = jnp.mean(
        jnp.sum(vmap(lambda _: vmap(logpx,(1, 0, None))(x, _, theta_x))(s), 1)
    )

    # compute KL[q(u)|p(u)]
    tr = jnp.trace(js.linalg.lu_solve(lu_fact, LK.T, trans=1).T)
    h = Kuu@KyyWTy
    logZ = 0.5*(jnp.dot(WTy.squeeze(), h)
                -jnp.linalg.slogdet(jnp.eye(L.shape[0])+LK)[1])
    KLqpu = -0.5*(tr+h.T@L@h)+WTy.T@h - logZ
    return Elogpx-KLqpu, s


def avg_neg_elbo(rng, theta, phi_n, logpx, cov_fn, x, t,
                 nsamples, precond, elbo_fn):
    """
    Calculate average negative elbo over training samples
    """
    T = t.shape[0]
    # unpack kernel params
    theta_x, theta_cov = theta[:2]
    N = theta_x[0][0][0].shape[0]
    theta_cov = tree_map(jnp.exp, theta_cov)
    # repeat in case the same kernel replicated for all ICs
    theta_cov = tree_map(lambda _: _.repeat(N-theta_cov[0].shape[0]+1),
                         theta_cov)
    # bound kernel params into something reasonable
    t_dist_mat = jnp.sqrt(squared_euclid_dist_mat(t))
    theta_cov = bound_se_kernel_params(
        theta_cov, sigma_min=1e-3,
        ls_min=jnp.min(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)]),
        ls_max=jnp.max(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)])
    )
    # calculate unscaled kernel (same for all samples in batch)
    # and update unscaled preconditioner
    K = K_TN_blocks(t, t, cov_fn, theta_cov, 1.)
    precond = fsai(K.swapaxes(1, 2).reshape(N*T, N*T), 2, 100, 1e-8,
                   precond, _identity)

    # compute elbo
    vlb, s = vmap(elbo_fn, (0, None, 0, None, None, 0, None, None, None, None))(
        jr.split(rng, x.shape[0]), theta, phi_n, logpx, cov_fn, x, t, nsamples,
        K, lax.stop_gradient(precond)
    )
    return -vlb.mean(), (s, precond)

avg_neg_tp_elbo = Partial(avg_neg_elbo, elbo_fn=structured_elbo)
avg_neg_gp_elbo = Partial(avg_neg_elbo, elbo_fn=gp_elbo)


if __name__ == "__main__":
    pdb.set_trace()
