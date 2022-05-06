import argparse
import jax.numpy as jnp
from jax import grad, vmap
from functools import partial
from .util import *
from .gamma import *
from .gaussian import *

# compute estimate of elbo terms that depend on r
def elbo_s(rng, theta, phi, logpx, K, x, y, r, nsamples):
    theta_K, theta_r, theta_x = theta
    phi_r, (What, yhat) = phi
    nu, rho = theta_K[0]*2, theta_K[1]*2
    K = lambda *_: K(*_, )*(nu-2)/(r*rho) # rescale kernel for given r
    Kxx = vmap(lambda t: vmap(lambda _: vmap(partial(K, t), in_axes=(None,0))(_, x))(x))(theta_K)
    # Assume diagonal What for now, so we have
    # What: (N,K)
    # yhat: (N,K)
    # ps_r: ((K,N), (K,N,N))
    ps_r = (What*yhat).T, -.5*(jnp.linalg.inv(Kxx) + vmap(jnp.diag, jnp.square(What).T))
    mu_s, Vs = gaussian_standardparams(ps_r)
    z, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,*mu_s.shape))
    # z: (nsamples,K,N)
    # y: (N,D)
    elbo = jnp.sum(gaussian_logZ(ps_r), 0) \
        - jnp.sum(mu_s.T*What*yhat) \
        - jnp.sum(-.5*What*vmap(jnp.diag, Vs).T) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, in_axes=(None,1,0)), in_axes=(None,0,None))(theta_x,z,y), 1), 0)
    return + mu_s*yhat 

# compute elbo estimate, assumes q(r) is gamma
def elbo(rng, theta, phi, K, x, y, nsamples):
    nsamples_r, nsamples_s = nsamples
    theta_K, theta_r, theta_x = theta
    phi_r, phi_s = phi
    r, rng = rngcall(lambda _: jax.random.gamma(_, phi_r[0], (nsamples, *phi_r[0].shape), rng)/phi_r[1])
    kl = gamma_kl(gamma_natparams_fromstandard(phi_r), gamma_natparams_fromstandard(theta_r))
    return jnp.mean(vmap(lambda _: elbo_s(rng, theta, phi, K, x, y, _, nsamples_s))(r), 0) - kl