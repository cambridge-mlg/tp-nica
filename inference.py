from jax._src.numpy.lax_numpy import trace
from jax._src.scipy.linalg import block_diag
from jax._src.scipy.special import gammaln
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
    naive_top_k,
    quad_form,
    fill_tril,
    jax_print,
    solve_precond_plus_diag,
    mbcg,
    _identity,
    fsai,
    krylov_subspace_sampling,
    lanczos_tridiag
)
from util import *
from gamma import *
from gaussian import *

from functools import partial
from time import perf_counter


def structured_elbo_s(key, theta, phi_s, logpx, x, t, tau, n_s_samples,
                      inference_args, K, G, P):
    theta_x, _ = theta[:2]
    L, h = phi_s
    N = h.shape[1]
    T = t.shape[0]
    max_nonzeros_P, P_iters, max_cg_iters, n_probe_vecs, kry_dim = inference_args

    # scale covariance
    K = K / tau[None, None, None, :]

    #1: compute parameters for \tilde{q(s|tau)}
    L = vmap(fill_tril, in_axes=(0, None))(L, N)
    L = vmap(lambda x: x.at[jnp.diag_indices_from(x)].set(
      jnp.exp(jnp.diag(x))))(L)
    L_inv = vmap(custom_tril_solve, (0, None))(L, jnp.eye(N))
    J_inv = jnp.matmul(L_inv.swapaxes(1, 2), L_inv)
    A = K.at[jnp.arange(T), jnp.arange(T)].add(J_inv).swapaxes(
      1, 2).reshape(N*T, N*T)
    K = K.swapaxes(1, 2).reshape(N*T, N*T)

    # scale preconditioner of inverse(cho_factor(K)) with current tau samples
    G = lax.stop_gradient(jnp.tile(jnp.sqrt(tau), T)[:, None]) * G

    # set up sampling
    key, key_z, key_u = jr.split(key, 3)
    Z = jr.normal(key_z, shape=(N*T, 2*n_s_samples + n_probe_vecs))
    K_mvp = lambda _: G@(K@(G.T@_))
    u0 = Z.T[:n_s_samples].reshape(-1, T, N)
    u0 = vmap(vmap(jnp.matmul), (None, 0))(L, u0)
    u1 = vmap(lambda _: krylov_subspace_sampling(key_u, K_mvp, _, kry_dim),
              in_axes=1, out_axes=1)(Z[:, n_s_samples:2*n_s_samples])
    u1 = (G.T@u1).T.reshape(-1, T, N)
    u = (u0+u1).reshape(n_s_samples, -1).T


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
    #P0 = jnp.eye(N*T)
    #P0 = P0.at[jnp.arange(N*T)[:, None], P[1]].set(P[0])
    P_val, _ = lax.stop_gradient(fsai(lax.stop_gradient(A), P_iters, max_nonzeros_P,
                                      1e-8, None, _identity))
    Minv_mvp = lambda b: jnp.matmul(P_val.T, jnp.matmul(P_val, b))
    #Minv_mvp = lambda b: jnp.matmul(P_val.T, vmap(lambda a, i:
    #                                          jnp.matmul(a[i],b[i]))(P_val, P_idx))
    #A_mvp2 = lambda _: P_val@A_mvp(P_val.T@_)

    # set up an run mbcg
    Z_tilde = custom_tril_solve(P_val, Z[:, 2*n_s_samples:])
    B = jnp.hstack((K@h.reshape(-1, 1), K@u, Z_tilde))
    solves, T_mats = mbcg(A_mvp, B, maxiter=max_cg_iters, M=Minv_mvp)
    #solves, T_mats = mbcg(A_mvp2, P_val@B, maxiter=max_cg_iters, M=None)
    #solves = P_val.T@solves

    # compute m 
    m = vmap(custom_choL_solve)(
        L, solves[:, 0].reshape(L.shape[0], -1)
    )

    # solve for sample and add mean
    s_zero = vmap(lambda x: vmap(custom_choL_solve)(L, x.reshape(T, N)), in_axes=1)(
        solves[:, 1:n_s_samples+1])
    s = m[None] + s_zero

    # compute likelihood expectation
    Elogpx = jnp.mean(
        jnp.sum(vmap(lambda _: vmap(logpx, (1, 0, None))(x, _, theta_x))(s), 1)
    )

    # compute logdet approximation
    ew, eV = jnp.linalg.eigh(T_mats[n_s_samples+1:])
    logdet_A_tilde = N*T * (jnp.log(ew) * eV[:, 0, :]**2).sum(1).mean()
    logdet_A = logdet_A_tilde - 2*jnp.log(jnp.diag(P_val)).sum()
    logdet_J = 2*jnp.log(jnp.einsum('ijj->ij', L)).sum()
    logdet = -logdet_A-logdet_J

    # compute trace approximation
    ste = vmap(jnp.dot, (1, 1))(solves[:, n_s_samples+1:],
                                K@(P_val.T@(P_val@Z_tilde))).mean()

    # compute mJm
    Lm = vmap(jnp.matmul)(L.swapaxes(1, 2), m)
    mJm = (Lm**2).sum()

    # compute KL & vlb
    KL = 0.5*((h*m).sum()-mJm-ste-logdet)
    vlb_s = Elogpx - KL
    #jax_print((Elogpx, KL))

    # "exact" validation
    #J = js.linalg.block_diag(*jnp.matmul(L, L.swapaxes(1, 2)))
    #m_x = jnp.linalg.solve(J+jnp.linalg.inv(K), h.reshape(-1))
    #solves2 = jnp.linalg.solve(jnp.linalg.inv(J)+K, K@h.reshape(-1))

    ####m_x2 = jnp.linalg.solve(J, solves[:,0])
    #m_x2 = jnp.linalg.solve(J, solves2)
    #tr_x = jnp.trace(jnp.linalg.solve(jnp.linalg.inv(J)+K, K))
    #mJm_x = jnp.dot(m_x, jnp.matmul(J, m_x))
    #logdet_A_x = jnp.linalg.slogdet(jnp.linalg.solve(jnp.linalg.inv(J)+K,
    #                                         jnp.linalg.inv(J)))[1]
    #hm = jnp.dot(m_x, h.reshape(-1))
    #kl_x = 0.5*(hm - tr_x - mJm_x - logdet_A_x)
    #jax_print((0, (h*m).sum(), mJm, ste, logdet))
    #jax_print((1, hm, mJm_x, tr_x, logdet_A_x))
    return vlb_s, s, P_val


# compute elbo estimate, assumes q(tau) is gamma
def structured_elbo(rng, theta, phi, logpx, x, t, nsamples, inference_args,
                    K, G, P):
    nsamples_s, nsamples_tau = nsamples
    theta_tau = theta[2]
    theta_tau = jnp.exp(theta_tau)
    phi_s, phi_tau = phi[:2]
    N = phi_tau[0].shape[0]
    max_nonzeros_P = inference_args[0]

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
    vlb_s, s, P_val = vmap(structured_elbo_s, (0, None, None, None, None,
             None, 0, None, None, None, None, None))(jr.split(rng, nsamples_tau),
                                         theta, phi_s, logpx, x, t, tau,
                                               nsamples_s, inference_args, K, G, P)

    # comput average preconditioner over tau samples
    P_avg = jnp.median(P_val, 0)
    P_idx = vmap(lambda _: naive_top_k(_, max_nonzeros_P)[1])(jnp.abs(P_avg))
    P_dense = P_avg[jnp.arange(P_avg.shape[0])[:, None], P_idx]
    return jnp.mean(vlb_s, 0) - kl, s, (P_dense, P_idx)


def gp_elbo(key, theta, phi_s, logpx, x, t, n_s_samples, inference_args,
            K, G, P):
    theta_x, _ = theta[:2]
    L, h = phi_s
    N = h.shape[1]
    T = t.shape[0]
    max_nonzeros_P, P_iters, max_cg_iters, n_probe_vecs, kry_dim  = inference_args

    #1: compute parameters for \tilde{q(s|tau)}
    L = vmap(fill_tril, in_axes=(0, None))(L, N)
    L = vmap(lambda x: x.at[jnp.diag_indices_from(x)].set(
      jnp.exp(jnp.diag(x))))(L)
    L_inv = vmap(custom_tril_solve, (0, None))(L, jnp.eye(N))
    J_inv = jnp.matmul(L_inv.swapaxes(1, 2), L_inv)
    A = K.at[jnp.arange(T), jnp.arange(T)].add(J_inv).swapaxes(
      1, 2).reshape(N*T, N*T)
    K = K.swapaxes(1, 2).reshape(N*T, N*T)

    # set up sampling
    key, key_z, key_u = jr.split(key, 3)
    Z = jr.normal(key_z, shape=(N*T, 2*n_s_samples + n_probe_vecs))
    K_mvp = lambda _: G@(K@(G.T@_))
    u0 = Z.T[:n_s_samples].reshape(-1, T, N)
    u0 = vmap(vmap(jnp.matmul), (None, 0))(L, u0)
    u1 = vmap(lambda _: krylov_subspace_sampling(key_u, K_mvp, _, kry_dim),
              in_axes=1, out_axes=1)(Z[:, n_s_samples:2*n_s_samples])
    u1 = (G.T@u1).T.reshape(-1, T, N)
    u = (u0+u1).reshape(n_s_samples, -1).T


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
    #P0 = jnp.eye(N*T)
    #P0 = P0.at[jnp.arange(N*T)[:, None], P[1]].set(P[0]) 

    #!! NOTICE NONE BELOW HARD-CODED FOR NOW!!
    P_val, P_idx = lax.stop_gradient(fsai(lax.stop_gradient(A), P_iters,
                                          max_nonzeros_P, 1e-8, None, _identity))
    Minv_mvp = lambda b: jnp.matmul(P_val.T, jnp.matmul(P_val, b))
    #Minv_mvp = lambda b: jnp.matmul(P_val.T, vmap(lambda a, i:
    #                                          jnp.matmul(a[i],b[i]))(P_val, P_idx))
    #A_mvp2 = lambda _: P_val@A_mvp(P_val.T@_)

    # save P for next iteration (turned off for now; see None input into fsai)
    P_dense = P_val[jnp.arange(P_val.shape[0])[:, None], P_idx]

    # set up an run mbcg
    Z_tilde = custom_tril_solve(P_val, Z[:, 2*n_s_samples:])
    B = jnp.hstack((K@h.reshape(-1, 1), K@u, Z_tilde))
    solves, T_mats = mbcg(A_mvp, B, maxiter=max_cg_iters, M=Minv_mvp)
    #solves, T_mats = mbcg(A_mvp2, P_val@B, maxiter=max_cg_iters, M=None)
    #solves = P_val.T@solves

    # compute m 
    m = vmap(custom_choL_solve)(
        L, solves[:, 0].reshape(L.shape[0], -1)
    )

    # solve for sample and add mean
    s_zero = vmap(lambda x: vmap(custom_choL_solve)(L, x.reshape(T, N)), in_axes=1)(
        solves[:, 1:n_s_samples+1])
    s = m[None] + s_zero

    # compute likelihood expectation
    Elogpx = jnp.mean(
        jnp.sum(vmap(lambda _: vmap(logpx, (1, 0, None))(x, _, theta_x))(s), 1)
    )

    # compute logdet approximation
    ew, eV = jnp.linalg.eigh(T_mats[n_s_samples+1:])
    logdet_A_tilde = N*T * (jnp.log(ew) * eV[:, 0, :]**2).sum(1).mean()
    logdet_A = logdet_A_tilde - 2*jnp.log(jnp.diag(P_val)).sum()
    logdet_J = 2*jnp.log(jnp.einsum('ijj->ij', L)).sum()
    logdet = -logdet_A-logdet_J

    # compute trace approximation
    ste = vmap(jnp.dot, (1, 1))(solves[:, n_s_samples+1:],
                                K@(P_val.T@(P_val@Z_tilde))).mean()

    # compute mJm
    Lm = vmap(jnp.matmul)(L.swapaxes(1, 2), m)
    mJm = (Lm**2).sum()

    # compute KL & vlb
    KL = 0.5*((h*m).sum()-mJm-ste-logdet)
    vlb_s = Elogpx - KL
    #jax_print((Elogpx, KL))

    # "exact" validation
    #J = js.linalg.block_diag(*jnp.matmul(L, L.swapaxes(1, 2)))
    #m_x = jnp.linalg.solve(J+jnp.linalg.inv(K), h.reshape(-1))
    #Cov = jnp.linalg.inv(J+jnp.linalg.inv(K))
    ##solves2 = jnp.linalg.solve(jnp.linalg.inv(J)+K, K@h.reshape(-1))

    ##m_x2 = jnp.linalg.solve(J, solves[:,0])
    ##m_x2 = jnp.linalg.solve(J, solves2)
    ###jax_print(jnp.max(jnp.abs((solves[:, 0]-solves2))).round(2))

    #tr_x = jnp.trace(jnp.linalg.solve(jnp.linalg.inv(J)+K, K))
    #mJm_x = jnp.dot(m_x, jnp.matmul(J, m_x))
    #logdet_A_x = jnp.linalg.slogdet(jnp.linalg.solve(jnp.linalg.inv(J)+K,
    #                                         jnp.linalg.inv(J)))[1]
    #kl_x = 0.5*(jnp.dot(m_x, h.reshape(-1)) - tr_x - mJm_x - logdet_A_x)
    return vlb_s, s, (P_dense, P_idx)


def avg_neg_elbo(rng, theta, phi_n, logpx, cov_fn, x, t,
                 nsamples, inference_args, G, P, elbo_fn):
    """
    Calculate average negative elbo over training samples
    """
    T = t.shape[0]
    max_nonzeros_G, G_iters = inference_args[:2]

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
    G, _ = fsai(lax.stop_gradient(K.swapaxes(1, 2).reshape(N*T, N*T)), G_iters,
                    max_nonzeros_G, 1e-8, lax.stop_gradient(G), _identity)

    # compute elbo
    vlb, s, P = vmap(elbo_fn, (0, None, 0, None, 0, None, None, None, None,
                               None, 0))(
        jr.split(rng, x.shape[0]), theta, phi_n, logpx, x, t, nsamples,
        inference_args[2:], K, lax.stop_gradient(G), P
    )
    return -vlb.mean(), (s, G, P)


avg_neg_tp_elbo = Partial(avg_neg_elbo, elbo_fn=structured_elbo)
avg_neg_gp_elbo = Partial(avg_neg_elbo, elbo_fn=gp_elbo)


if __name__ == "__main__":
    pdb.set_trace()
