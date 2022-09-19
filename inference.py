import argparse
import jax
from jax._src.numpy.linalg import slogdet
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import numpy as np
import pdb
from jax import grad, value_and_grad, vmap, jit
from jax.lax import scan
from jax.random import split
from jax.tree_util import tree_map, Partial

from functools import partial
from kernels import (
    se_kernel_fn,
    bound_se_kernel_params,
    squared_euclid_dist_mat
)
from data_generation import sample_tprocess
from utils import custom_solve, jax_print, comp_K_N, fill_triu, reorder_covmat
from util import *
from gamma import *
from gaussian import *


def structured_elbo_s(rng, theta, phi_s, logpx, cov_fn, x, t, tau, nsamples):
    theta_x, theta_cov = theta[:2]
    theta_cov = tree_map(lambda _: jnp.exp(_), theta_cov)
    t_dist_mat = jnp.sqrt(squared_euclid_dist_mat(t))
    theta_cov = bound_se_kernel_params(
        theta_cov, sigma_min=1e-3,
        ls_min=jnp.min(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)]),
        ls_max=jnp.max(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)])
    )
    What, yhat, tu = phi_s
    N, n_pseudo = yhat.shape
    T = t.shape[0]
    Kuu = vmap(lambda b: vmap(lambda a:
        comp_K_N(a, b, cov_fn, theta_cov)/tau[:, None]
    )(tu))(tu)
    Kuu = Kuu.swapaxes(1, 2).reshape(n_pseudo*N, n_pseudo*N)
    Ksu = vmap(lambda b:vmap(lambda a:
        comp_K_N(a, b, cov_fn, theta_cov)/tau[:, None])(tu))(t)
    Ksu = Ksu.swapaxes(1, 2).reshape(T*N, n_pseudo*N)
    kss = vmap(
        lambda tc: vmap(lambda t: cov_fn(t, t, tc))(t)
        )(theta_cov) / tau[:, None]

    # compute parameters for \tilde{q(s|tau)}
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

    # compute E_{\tilde{q(s|tau)}}[log_p(x_t|s_t)]
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


# compute elbo estimate, assumes q(tau) is gamma
def structured_elbo(rng, theta, phi, logpx, cov_fn, x, t, nsamples):
    nsamples_s, nsamples_tau = nsamples
    theta_tau = theta[2]
    theta_tau = jnp.exp(theta_tau)
    phi_s, phi_tau = phi[:2]
    tau, rng = rngcall(gamma_sample, rng, gamma_natparams_fromstandard(phi_tau),
                       (nsamples_tau, *phi_tau[0].shape))
    kl = jnp.sum(
        gamma_kl(
            gamma_natparams_fromstandard(phi_tau),
            gamma_natparams_fromstandard((theta_tau/2, theta_tau/2))), 0)
    vlb_s, s = vmap(lambda _: structured_elbo_s(
        rng, theta, phi_s, logpx, cov_fn, x, t, _, nsamples_s))(tau)
    return jnp.mean(vlb_s, 0) - kl, s


def gp_elbo(rng, theta, phi_s, logpx, cov_fn, x, t, nsamples):
    theta_x, theta_cov = theta[:2]
    theta_cov = tree_map(lambda _: jnp.exp(_), theta_cov)
    t_dist_mat = jnp.sqrt(squared_euclid_dist_mat(t))
    theta_cov = bound_se_kernel_params(
        theta_cov, sigma_min=1e-3,
        ls_min=jnp.min(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)]),
        ls_max=jnp.max(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)])
    )
    What, yhat, tu = phi_s
    N, n_pseudo = yhat.shape
    T = t.shape[0]
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


def avg_neg_elbo(rng, theta, phi_n, logpx, cov_fn, x, t, nsamples, elbo_fn):
    """
    Calculate average negative elbo over training samples
    """
    vlb, s = vmap(lambda a, b, c: elbo_fn(
        a, theta, b, logpx, cov_fn, c, t, nsamples))(jr.split(rng, x.shape[0]),
                                                     phi_n, x)
    return -vlb.mean(), s


avg_neg_tp_elbo = Partial(avg_neg_elbo, elbo_fn=structured_elbo)
avg_neg_gp_elbo = Partial(avg_neg_elbo, elbo_fn=gp_elbo)


if __name__ == "__main__":
    pdb.set_trace()
