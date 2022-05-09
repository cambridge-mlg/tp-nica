import argparse
import jax
import jax.numpy as jnp
from jax import grad, vmap
from functools import partial
from tprocess.kernels import SE_kernel
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
    rcov = lambda rfoo, tr, tc, x1, x2: cov(tc, x1, x2)*(tr[0]*2-2)/(rfoo*tr[1]*2) # rescale kernel for given r
    tree_first = lambda _: tree_map(lambda _: _[0], _)
    print(theta_r)
    print(theta_cov)
    print(rcov(r[0], tree_first(theta_r), tree_first(theta_cov), x[0], x[2]))
    print(r[0].shape)
    print(tree_first(theta_r))
    print(tree_first(theta_cov))
    print(x[0].shape)
    print(x[1].shape)
    print("**********************")
    Kxx = vmap(lambda r:
        vmap(lambda tr, tc: \
            vmap(lambda x1: 
                vmap(partial(rcov, r, tr, tc, x1)))(x))(theta_r, theta_cov))(r)
    print(x.shape)
    # Assume diagonal What for now, so we have
    # What: (N,K)
    # yhat: (N,K)
    # ps_r: ((K,N), (K,N,N))
    ps_r = (What*yhat).T, -.5*(jnp.linalg.inv(Kxx) + vmap(jnp.diag, jnp.square(What).T))
    mu_s, Vs = gaussian_standardparams(ps_r)
    print(Vs.shape)
    print(ps_r[1].shape)
    s, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,*mu_s.shape))
    # z: (nsamples,K,N)
    # y: (N,D)
    elbo = jnp.sum(gaussian_logZ(ps_r), 0) \
        - jnp.sum(mu_s.T*What*yhat) \
        - jnp.sum(-.5*What*vmap(jnp.diag, Vs).T) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, in_axes=(None,0,0)), in_axes=(None,0,None))(theta_x,s,y), 1), 0)
    return elbo

# compute elbo estimate, assumes q(r) is gamma
def elbo(rng, theta, phi, logpx, cov, x, y, nsamples):
    nsamples_r, nsamples_s = nsamples
    theta_cov, theta_r, theta_x = theta
    phi_r, phi_s = phi
    r, rng = rngcall(lambda _: jax.random.gamma(_, phi_r[0], (nsamples_r, *phi_r[0].shape))/phi_r[1], rng)
    kl = gamma_kl(gamma_natparams_fromstandard(phi_r), gamma_natparams_fromstandard(theta_r))
    return jnp.mean(vmap(lambda _: elbo_s(rng, theta, phi, logpx, cov, x, y, _, nsamples_s))(r), 0) - kl

def main():
    rng = jax.random.PRNGKey(0)
    K, N = 3, 20
    cov = squaredexp
    logpx = lambda _, s, y: y*jnp.sum(s, 0) - jnp.exp(jnp.sum(s, 0)) - jnp.log(jax.scipy.special.factorial(y))
    theta_cov = jnp.ones(K)*1.0, jnp.ones(K)*.05
    theta_r = jnp.ones(K)*2, jnp.ones(K)*2
    theta_x = ()
    phi_r = jnp.ones(K), jnp.ones(K)
    phi_s, rng = rngcall(lambda _: (jnp.zeros((K,N,100,3)), jax.random.normal(_, ((100,3)))), rng)
    theta = theta_cov, theta_r, theta_x
    phi = phi_r, phi_s
    nsamples = (10, 5)
    x = jnp.linspace(0, 10, 100)
    print("hello")
    s, rng = rngcall(sample_multiple_tprocesses, rng, x, [lambda _: _*0 for _ in range(K)], [SE_kernel(theta_cov) for _ in range(K)], list(zip(*theta_r)))
    print(jnp.exp(jnp.sum(s, 0)).shape)
    print(rng)
    y, rng = rngcall(jax.random.poisson, rng, jnp.exp(jnp.sum(s, 0)), (100,))
    print(y.shape)
    print(elbo(rng, theta, phi, logpx, cov, x, y, nsamples))


if __name__ == "__main__":
    main()