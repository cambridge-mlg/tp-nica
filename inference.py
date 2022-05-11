import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
import pdb
from jax import grad, value_and_grad, vmap, jit
from jax.lax import scan
from jax.random import split
from functools import partial
from math import factorial
from tprocess.kernels import se_kernel_fn, compute_K
from tprocess.sampling import sample_tprocess
from util import *
from gamma import *
from gaussian import *

# compute estimate of elbo terms that depend on rinv
def elbo_s(rng, theta, phi_s, logpx, cov, x, t, rinv, nsamples):
    theta_x, theta_cov = theta[:2]
    What, yhat = phi_s
    Ktt = compute_K(t, cov, theta_cov).transpose(2,0,1)/rinv[:,None,None]
    # Assume diagonal What for now, so we have
    # What: (N,T)
    # yhat: (N,T)
    # ps_r: ((N,T), (N,T,T))
    qs_r = (What*yhat), -.5*(jnp.linalg.inv(Ktt) + vmap(jnp.diag)(jnp.square(What)))
    mu_s, Vs = gaussian_standardparams(qs_r)
    s, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,mu_s.shape[0]))
    # s: (nsamples,N,T)
    # x: (M,T)
    elbo = jnp.sum(gaussian_logZ2(qs_r), 0) \
        - jnp.sum(mu_s*What*yhat) \
        - jnp.sum(-.5*jnp.square(What)*(vmap(jnp.diag)(Vs) + jnp.square(mu_s))) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, (1, 1, None)),
                                (None, 0, None))(x, s, theta_x), 1), 0)
    return elbo, s


# compute elbo estimate, assumes q(r) is gamma
def elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
    nsamples_s, nsamples_r = nsamples
    theta_r = theta[2]
    phi_s, phi_r = phi
    rinv, rng = rngcall(gamma_sample, rng, gamma_natparams_fromstandard(phi_r), (nsamples_r, *phi_r[0].shape))
    kl = jnp.sum(
        gamma_kl(
            gamma_natparams_fromstandard(phi_r),
            gamma_natparams_fromstandard((theta_r/2, theta_r/2))), 0)
    vlb_r, s = vmap(lambda _: elbo_s(rng, theta, phi_s, logpx, cov, x, t, _, nsamples_s))(rinv)
    return jnp.mean(vlb_r, 0) - kl, s


def avg_neg_elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
    """
    Calculate average negative elbo over training samples
    """
    elbo, s = vmap(
        lambda a, b, c: elbo(a, theta, b, logpx, cov, c, t, nsamples)
    )(jr.split(rng, x.shape[0]), phi, x)
    return -elbo.mean(), s


# compute elbo over multiple training examples
def main():
    rng = jax.random.PRNGKey(0)
    N, T = 1, 20
    cov = se_kernel_fn
    noisesd = .2
    logpx = lambda _, s, x: jax.scipy.stats.norm.logpdf(x.reshape(()), jnp.sum(s, 0), noisesd)
    theta_cov = jnp.ones(N)*1.0, jnp.ones(N)*1.0
    theta_r = jnp.ones(N)*4.0
    theta_x = () # likelihood parameters
    phi_r = jnp.ones(N)*5, jnp.ones(N)*5
    phi_s, rng = rngcall(lambda _: (jnp.zeros((N,T)), jr.normal(_, ((N,T)))*.05), rng)
    theta = theta_x, theta_cov, theta_r
    phi = phi_s, phi_r
    nsamples = (5, 10) # (nssamples, nrsamples)
    t = jnp.linspace(0, 100, T)
    (s, rinv), rng = rngcall(lambda k: \
        vmap(lambda a, b, c: sample_tprocess(a, t, lambda _: _*0, cov, b, c))(
            split(k, N), theta_cov, theta_r),
        rng)
    x, rng = rngcall(lambda k: jax.random.normal(k, (1, T))*noisesd + jnp.sum(s, 0), rng)
    lr = 1e-4
    print(f"ground truth rinv: {rinv}")
    def step(phi, rng):
        vlb, g = value_and_grad(elbo, 2)(rng, theta, phi, logpx, cov, x, t, nsamples)
        return tree_add(phi, tree_scale(g, lr)), vlb
    nepochs = 100000
    for i in range(1, nepochs):
        rng, key = split(rng)
        phi, vlb = scan(step, phi, split(key, 1000))
        phi_r_natparams = gamma_natparams_fromstandard(phi[1])
        print(f"{i}: elbo={vlb.mean()}, E[rinv]={gamma_mean(phi_r_natparams)}, V[rinv]={gamma_var(phi_r_natparams)}")

if __name__ == "__main__":
    # print("foo")
    # phi1 = jnp.array(4.0), jnp.array(7.0)
    # phi2 = jnp.array(20.0), jnp.array(15.0)
    # m = (20/2)/(15/2)
    # print(m)
    # def loss(key, phi1, phi2):
    #     a, b = phi1[0]/2, phi1[1]/2
    #     x = jax.random.gamma(key, a, (100,))/b
    #     return jnp.mean(gamma_logprob(gamma_natparams_fromweird(phi1), x) - gamma_logprob(gamma_natparams_fromweird(phi2), x))
    # rng = jax.random.PRNGKey(0)
    # for _ in range(100000):
    #     rng, key = split(rng)
    #     kl, g = value_and_grad(loss, argnums=1)(key, phi1, phi2)
    #     phi1 = tree_sub(phi1, tree_scale(g, 1e-3))
    #     if _ % 100 == 0:
    #         print(f"{_}: {kl}. {gamma_kl(gamma_natparams_fromweird(phi1), gamma_natparams_fromweird(phi2))}, {gamma_mean(gamma_natparams_fromweird(phi1))}")
    main()
