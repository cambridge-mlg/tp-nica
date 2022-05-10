import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import value_and_grad, vmap
from jax.lax import scan
from jax.random import split
from functools import partial
from math import factorial
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
def elbo_s(rng, theta, phi, logpx, cov, t, x, r, nsamples):
    theta_cov, theta_r, theta_t = theta
    (What, yhat), phi_r = phi
    scale = (theta_r[0]*2-2)/(r*theta_r[1]*2)
    Ktt = vmap(lambda tc: \
            vmap(lambda t1: 
                vmap(lambda t2:
                    cov(t1, t2, tc)
                )(t)
            )(t)
        )(theta_cov)*scale[:,None,None]
    # Assume diagonal What for now, so we have
    # What: (N,T)
    # yhat: (N,T)
    # ps_r: ((N,T), (N,T,T))
    ps_r = (What*yhat), -.5*(jnp.linalg.inv(Ktt) + vmap(jnp.diag)(jnp.square(What)))
    mu_s, Vs = gaussian_standardparams(ps_r)
    s, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,mu_s.shape[0]))
    # s: (nsamples,N,T)
    # y: (M,T)
    elbo = jnp.sum(gaussian_logZ(ps_r), 0) \
        - jnp.sum(mu_s*What*yhat) \
        - jnp.sum(-.5*What.T*vmap(jnp.diag)(Vs.T)) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, (None,1,1)), (None,0,None))(theta_t,s,x), 1), 0)
    return elbo


# compute elbo estimate, assumes q(r) is gamma
def elbo(rng, theta, phi, logpx, cov, t, x, nsamples):
    nsamples_r, nsamples_s = nsamples
    theta_cov, theta_r, theta_t = theta
    phi_s, phi_r = phi
    r, rng = rngcall(lambda _: jax.random.gamma(_, phi_r[0], (nsamples_r, *phi_r[0].shape))/phi_r[1], rng)
    kl = jnp.sum(gamma_kl(gamma_natparams_fromstandard(phi_r), gamma_natparams_fromstandard(theta_r)), 0)
    return jnp.mean(vmap(lambda _: elbo_s(rng, theta, phi, logpx, cov, t, x, _, nsamples_s))(r), 0) - kl


def main():
    rng = jax.random.PRNGKey(0)
    N, T = 3, 20
    cov = se_kernel_fn
    logpx = lambda _, s, x: jax.scipy.stats.poisson.logpmf(x.reshape(()), jnp.exp(jnp.sum(s, 0)))
    theta_cov = jnp.ones(N)*1.0, jnp.ones(N)*.05
    theta_r = jnp.ones(N)*2, jnp.ones(N)*2
    theta_x = () # likelihood parameters
    phi_r = jnp.ones(N)*2, jnp.ones(N)*2
    phi_s, rng = rngcall(lambda _: (jnp.zeros((N,T)), jax.random.normal(_, ((N,T)))), rng)
    theta = theta_cov, theta_r, theta_x
    phi = phi_s, phi_r
    nsamples = (10, 5) # (nrsamples, nssamples)
    t = jnp.linspace(0, 10, T)
    (s, r), rng = rngcall(sample_multiple_tprocesses, rng, t, [lambda _: _*0 for _ in range(N)], [cov for _ in range(N)], [(1.0, .05) for _ in range(N)], list(zip(*theta_r)))
    x, rng = rngcall(jax.random.poisson, rng, jnp.exp(jnp.sum(s, 0)), (1,T))
    lr = 1e-4
    print(f"ground truth r: {r}")
    def step(phi, rng):
        vlb, g = value_and_grad(elbo, 2)(rng, theta, phi, logpx, cov, t, x, nsamples)
        return tree_add(phi, tree_scale(g, lr)), vlb
    nepochs = 10000
    for i in range(1, nepochs):
        rng, key = split(rng)
        phi, vlb = scan(step, phi, split(key, 100))
        print(f"{i}: elbo={vlb.mean()}, E[r]={gamma_meanparams(gamma_natparams_fromstandard(phi[1]))[0]}")

if __name__ == "__main__":
    main()
