import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import pdb
from jax import value_and_grad, vmap
from jax.lax import scan
from jax.random import split
from functools import partial
from math import factorial
from tprocess.kernels import se_kernel_fn
from tprocess.sampling import sample_tprocess
from util import *
from gamma import *
from gaussian import *

# compute estimate of elbo terms that depend on rinv
def elbo_s(rng, theta, phi, logpx, cov, t, x, rinv, nsamples):
    theta_cov, theta_r, theta_x = theta
    (What, yhat), phi_r = phi
    nu, rho = theta_r 
    scale = (nu-2)/(rinv*rho)
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
    qs_r = (What*yhat), -.5*(jnp.linalg.inv(Ktt) + vmap(jnp.diag)(jnp.square(What)))
    mu_s, Vs = gaussian_standardparams(qs_r)
    s, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,mu_s.shape[0]))
    # s: (nsamples,N,T)
    # y: (M,T)
    elbo = jnp.sum(gaussian_logZ2(qs_r), 0) \
        - jnp.sum(mu_s*What*yhat) \
        - jnp.sum(-.5*jnp.square(What)*(vmap(jnp.diag)(Vs) + jnp.square(mu_s))) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, (None,1,1)), (None,0,None))(theta_x,s,x), 1), 0)
    return elbo, s

def gamma_natparams_fromweird(x):
    return 2*x[0]-1, -2*x[1]

# compute elbo estimate, assumes q(r) is gamma
def elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
    nsamples_s, nsamples_r = nsamples
    theta_cov, theta_r, theta_x = theta
    phi_s, phi_r = phi
    nu, rho = phi_r
    rinv, rng = rngcall(lambda _: jax.random.gamma(_, nu/2, (nsamples_r, *phi_r[0].shape))/(rho/2), rng)
    kl = jnp.sum(gamma_kl(gamma_natparams_fromweird(phi_r), gamma_natparams_fromweird(theta_r)), 0)
    vlb_r, s = vmap(lambda _: elbo_s(rng, theta, phi, logpx, cov, t, x, _, nsamples_s))(rinv)
    return jnp.mean(vlb_r, 0) - kl, s

# compute elbo over multiple training examples
def main():
    rng = jax.random.PRNGKey(0)
    N, T = 1, 50
    cov = se_kernel_fn
    logpx = lambda _, s, x: jax.scipy.stats.poisson.logpmf(x.reshape(()), jnp.exp(jnp.sum(s, 0)))
    theta_cov = jnp.ones(N)*1.0, jnp.ones(N)*1.0
    theta_r = jnp.ones(N)*5, jnp.ones(N)*3
    theta_x = () # likelihood parameters
    phi_r = jnp.ones(N)*20, jnp.ones(N)*3
    phi_s, rng = rngcall(lambda _: (jnp.zeros((N,T)), jr.normal(_, ((N,T)))), rng)
    theta = theta_cov, theta_r, theta_x
    phi = phi_s, phi_r
    nsamples = (5, 10) # (nssamples, nrsamples)
    t = jnp.linspace(0, 100, T)
    (s, rinv), rng = rngcall(lambda k: \
        vmap(lambda a, b, c: sample_tprocess(a, t, lambda _: _*0, cov, b, c))(
            split(k, N), theta_cov, theta_r),
        rng)
    x, rng = rngcall(jax.random.poisson, rng, jnp.exp(jnp.sum(s, 0)), (1, T))
    lr = 1e-4
    print(f"ground truth rinv: {rinv}")
    def step(phi, rng):
        (vlb, s), g = value_and_grad(elbo, 2, has_aux=True)(rng, theta, phi, logpx, cov, x, t, nsamples)
        return tree_add(phi, tree_scale(g, lr)), s
    nepochs = 1000
    print(s[0])
    for i in range(1, nepochs):
        rng, key = split(rng)
        phi, s_samples = scan(step, phi, split(key, 1000))
        phi_natparams = gamma_natparams_fromweird(phi[1])
        print(f"{i}: {jnp.corrcoef(s_samples.mean(axis=(0,1,2,3)), s[0])[0,1]}, E[rinv]={gamma_mean(phi_natparams)}, V[rinv]={gamma_var(phi_natparams)}")
        # print(f"{i}: elbo={vlb.mean()}, E[rinv]={gamma_mean(phi_natparams)}, V[rinv]={gamma_var(phi_natparams)}")
    print(s_samples.mean(axis=(0,1,2,3)))
    print(s[0])

if __name__ == "__main__":
    main()
