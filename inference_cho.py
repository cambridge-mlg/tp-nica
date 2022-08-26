import argparse
import jax

from jax._src.numpy.linalg import slogdet
from jax._src.scipy.sparse.linalg import cg

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import numpy as np
import pdb

from jax import grad, value_and_grad, vmap, jit
from jax.lax import scan
from jax.random import split
from jax.tree_util import tree_map

from functools import partial
from math import factorial
from kernels import (
    se_kernel_fn,
    bound_se_kernel_params,
    squared_euclid_dist_mat
)
from data_generation import sample_tprocess
from utils import (
    custom_chol_solve,
    custom_tril_solve,
    jax_print,
    comp_K_N,
    fill_triu
)
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
    L_blocks = jnp.moveaxis(jnp.einsum('ijk, ilk->jlk', What, What), -1, 0)
    L = js.linalg.block_diag(*L_blocks)
    Linv = js.linalg.block_diag(
        *vmap(custom_chol_solve, in_axes=(0, None, (-1, None)))(
        L_blocks, jnp.eye(N), (What, False))
    )
    cho_fact = js.linalg.cho_factor(Kuu+Linv)
    #KyyLinv = js.linalg.cho_solve(cho_fact, Linv)
    KyyLinv = custom_chol_solve(Kuu+Linv, Linv, cho_fact)
    KyyLinvWTy = KyyLinv @ WTy
    mu_s = Ksu @ KyyLinvWTy
    #cov_s = vmap(lambda X, z: jnp.diag(z)-X@js.linalg.cho_solve(cho_fact, X.T)
    #             , in_axes=(0, -1)
    #    )(Ksu.reshape(T, N, -1), kss)
    cov_solve = vmap(lambda b: custom_tril_solve(Linv[0].T, b.T))(
            Ksu.reshape(T, N, -1))
    cov_s = vmap(lambda X, z: jnp.diag(z)-X.T@X,
                 in_axes=(0, -1))(cov_solve, kss)
    jax_print(jnp.min(jnp.linalg.eigh(cov_s)[0]))
    s, rng = rngcall(lambda _: jr.multivariate_normal(_, mu_s.reshape(T, N),
        cov_s, shape=(nsamples, T)), rng)

    # compute E_{\tilde{q(s|tau)}}[log_p(x_t|s_t)]
    Elogpx = jnp.mean(
        jnp.sum(vmap(lambda _: vmap(logpx,(1, 0, None))(x, _, theta_x))(s), 1)
    )

    # compute KL[q(u)|p(u)]
    tr = jnp.trace(L @ Kuu @ KyyLinv)
    h = Kuu@KyyLinvWTy
    logZ = 0.5*(jnp.dot(WTy.squeeze(), h)
                -jnp.linalg.slogdet(jnp.eye(L.shape[0])+L@Kuu)[1])
    KLqpu = -0.5*(tr+h.T@L@h)+WTy.T@h - logZ
    #jax_print(Elogpx)
    #jax_print(KLqpu)
    return Elogpx-KLqpu, s


# compute elbo estimate, assumes q(tau) is gamma
def structured_elbo(rng, theta, phi, logpx, cov_fn, x, t, nsamples):
    nsamples_s, nsamples_tau = nsamples
    theta_tau = theta[2]
    theta_tau = jnp.exp(theta_tau)
    phi_s, phi_tau = phi[:2]
    tau, rng = rngcall(gamma_sample, rng, gamma_natparams_fromstandard(phi_tau),
                       (nsamples_tau, *phi_tau[0].shape))
    #jax_print(theta_tau/2)
    kl = jnp.sum(
        gamma_kl(
            gamma_natparams_fromstandard(phi_tau),
            gamma_natparams_fromstandard((theta_tau/2, theta_tau/2))), 0)
    vlb_s, s = vmap(lambda _: structured_elbo_s(
        rng, theta, phi_s, logpx, cov_fn, x, t, _, nsamples_s))(tau)
    #jax_print(kl)
    return jnp.mean(vlb_s, 0) - kl, s


def avg_neg_elbo(rng, theta, phi_n, logpx, cov_fn, x, t, nsamples):
    """
    Calculate average negative elbo over training samples
    """
    vlb, s = vmap(lambda a, b, c: structured_elbo(
        a, theta, b, logpx, cov_fn, c, t, nsamples))(jr.split(rng, x.shape[0]),
                                                     phi_n, x)
    return -vlb.mean(), s


def elbo_main():
    rng = jax.random.PRNGKey(0)
    N, T = 5, 20
    cov = se_kernel_fn
    noisesd = .5
    logpx = lambda _, s, x: jax.scipy.stats.norm.logpdf(x.reshape(()), jnp.sum(s, 0), noisesd)
    theta_cov = jnp.ones(N)*1.0, jnp.ones(N)*1.0
    theta_tau = jnp.ones(N)*4.0
    theta_x = () # likelihood parameters
    t = jnp.linspace(0, 10, T)[:,None]
    (s, tau), rng = rngcall(lambda k: \
        vmap(lambda a, b, c: sample_tprocess(a, t, lambda _: .0, cov, b, c))(
            split(k, N), theta_cov, theta_tau),
        rng)

    npseudopoints = T//2
    tu, rng = rngcall(lambda k: jax.random.uniform(k, shape=(npseudopoints,), minval=jnp.min(t), maxval=jnp.max(t))[:,None], rng)
    phi_tau = jnp.ones(N)*5, jnp.ones(N)*5
    phi_s, rng = rngcall(lambda _: (jnp.zeros((N,len(tu))), jr.normal(_, ((N,len(tu))))*.05, tu), rng)
    theta = theta_x, theta_cov, theta_tau
    phi = phi_s, phi_tau
    nsamples = (5, 10) # (nssamples, nrsamples)
    x, rng = rngcall(lambda k: jax.random.normal(k, (1, T))*noisesd + jnp.sum(s, 0), rng)
    lr = 1e-3
    print(f"ground truth tau: {tau}")
    def step(phi, rng):
        vlb, g = value_and_grad(structured_elbo, 2)(rng, theta, phi, logpx, cov, x, t, nsamples)
        return tree_add(phi, tree_scale(g, lr)), vlb
    nepochs = 100000
    for i in range(1, nepochs):
        rng, key = split(rng)
        phi, vlb = scan(step, phi, split(key, 10000))
        phi_tau_natparams = gamma_natparams_fromstandard(phi[1])
        print(f"{i}: elbo={vlb.mean(0)}, E[tau]={gamma_mean(phi_tau_natparams)}, V[tau]={gamma_var(phi_tau_natparams)}")

if __name__ == "__main__":
    elbo_main()
