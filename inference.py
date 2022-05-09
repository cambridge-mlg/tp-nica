import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap
from jax.random import split
from functools import partial
from tprocess.kernels import se_kernel_fn
from tprocess.sampling import sample_multiple_tprocesses
from util import *
from gamma import *
from gaussian import *

def euclid_dist(x, y):
    return jnp.linalg.norm(x-y)

def squaredexp(params, x, y):
    sigma, lscale = params
    return sigma**2 * jnp.exp(-0.5*euclid_dist(x, y)**2 / lscale**2)

# compute estimate of elbo terms that depend on r
def elbo_s(rng, theta, phi, logpx, cov, x, y, r, nsamples):
    theta_cov, theta_r, theta_x = theta
    phi_r, (What, yhat) = phi
    cov_s = lambda tc, x1, x2: cov(tc, x1, x2)
    tree_first = lambda _: tree_map(lambda _: _[0], _)
    scale = (theta_r[0]*2-2)/(r*theta_r[1]*2)
    Kxx = vmap(lambda tc: \
        vmap(lambda x1: 
            vmap(partial(cov_s, tc, x1))(x))(x)
        )(theta_cov)*scale[:,None,None]
    # Assume diagonal What for now, so we have
    # What: (N,K)
    # yhat: (N,K)
    # ps_r: ((K,N), (K,N,N))
    ps_r = (What*yhat).T, -.5*(jnp.linalg.inv(Kxx) + vmap(jnp.diag)(jnp.square(What).T))
    mu_s, Vs = gaussian_standardparams(ps_r)
    s, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,mu_s.shape[0]))
    # z: (nsamples,K,N)
    # y: (N,D)
    elbo = jnp.sum(gaussian_logZ(ps_r), 0) \
        - jnp.sum(mu_s.T*What*yhat) \
        - jnp.sum(-.5*What*vmap(jnp.diag)(Vs.T)) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, (None,1,0)), (None,0,None))(theta_x,s,y), 1), 0)
    return elbo

# compute elbo estimate, assumes q(r) is gamma
def elbo(rng, theta, phi, logpx, cov, x, y, nsamples):
    nsamples_r, nsamples_s = nsamples
    theta_cov, theta_r, theta_x = theta
    phi_r, phi_s = phi
    r, rng = rngcall(lambda _: jax.random.gamma(_, phi_r[0], (nsamples_r, *phi_r[0].shape))/phi_r[1], rng)
    kl = jnp.sum(gamma_kl(gamma_natparams_fromstandard(phi_r), gamma_natparams_fromstandard(theta_r)), 0)
    return jnp.mean(vmap(lambda _: elbo_s(rng, theta, phi, logpx, cov, x, y, _, nsamples_s))(r), 0) - kl

def main():
    rng = jax.random.PRNGKey(0)
    K, N = 3, 20
    cov = squaredexp
    logpx = lambda _, s, y: y*jnp.sum(s, 0) - jnp.exp(jnp.sum(s, 0)) # - jnp.log(jax.scipy.special.factorial(y))
    theta_cov = jnp.ones(K)*1.0, jnp.ones(K)*.05
    theta_r = jnp.ones(K)*2, jnp.ones(K)*2
    theta_x = ()
    phi_r = jnp.ones(K), jnp.ones(K)
    phi_s, rng = rngcall(lambda _: (jnp.zeros((N,K)), jax.random.normal(_, ((N,K)))), rng)
    theta = theta_cov, theta_r, theta_x
    phi = phi_r, phi_s
    nsamples = (10, 5)
    x = jnp.linspace(0, 10, N)
    s, rng = rngcall(sample_multiple_tprocesses, rng, x, [lambda _: _*0 for _ in range(K)], [se_kernel_fn for _ in range(K)], [(1.0, .05) for _ in range(K)], list(zip(*theta_r)))
    y, rng = rngcall(jax.random.poisson, rng, jnp.exp(jnp.sum(s, 0)), (N,))
    print(elbo(rng, theta, phi, logpx, cov, x, y, nsamples))


if __name__ == "__main__":
    main()
    r, rng = rngcall(lambda _: jr.gamma(_, phi_r[0], (nsamples, *phi_r[0].shape), rng)/phi_r[1])
    kl = gamma_kl(gamma_natparams_fromstandard(phi_r), gamma_natparams_fromstandard(theta_r))
    return jnp.mean(vmap(lambda _: elbo_s(rng, theta, phi, K, x, y, _, nsamples_s))(r), 0) - kl
