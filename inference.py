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

# compute estimate of elbo terms that depend on tau
def elbo_s(rng, theta, phi_s, logpx, cov, x, t, tau, nsamples):
    theta_x, theta_cov = theta[:2]
    What, yhat = phi_s
    Kinv = jnp.linalg.inv(compute_K(t, cov, theta_cov).transpose(2,0,1))*tau[:,None,None]
    # Assume diagonal What for now, so we have
    # What: (N,T)
    # yhat: (N,T)
    # ps_tau: ((N,T), (N,T,T))
    qs_tau = (What*yhat), -.5*(Kinv + vmap(jnp.diag)(jnp.square(What)))
    mu_s, Vs = gaussian_standardparams(qs_tau)
    s = gaussian_sample(rng, qs_tau, (nsamples,))
#    s, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,mu_s.shape[0]))
    # s: (nsamples,N,T)
    # x: (M,T)
    T = x.shape[1]
    elbo = jnp.sum(gaussian_logZ2(qs_tau), 0) \
        - jnp.sum(mu_s*What*yhat) \
        - jnp.sum(-.5*jnp.square(What)*(vmap(jnp.diag)(gaussian_meanparams(qs_tau)[1]))) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, (None,1,1)), (None,0,None))(theta_x,s,x), 1), 0) \
        + jnp.sum(.5*jnp.linalg.slogdet(Kinv/(2*jnp.pi))[1], 0)
    return elbo


# compute elbo estimate, assumes q(r) is gamma
def elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
    nsamples_s, nsamples_tau = nsamples
    theta_tau = theta[2]
    phi_s, phi_tau = phi
    tau, rng = rngcall(gamma_sample, rng, gamma_natparams_fromstandard(phi_tau), (nsamples_tau, *phi_tau[0].shape))
    kl = jnp.sum(
        gamma_kl(
            gamma_natparams_fromstandard(phi_tau),
            gamma_natparams_fromstandard((theta_tau/2, theta_tau/2))), 0)
    vlb_s = vmap(lambda _: elbo_s(rng, theta, phi_s, logpx, cov, x, t, _, nsamples_s))(tau)
    return jnp.mean(vlb_s, 0) - kl


def avg_neg_elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
    """
    Calculate average negative elbo over training samples
    """
    vlb, s = vmap(
        lambda a, b, c: elbo(a, theta, b, logpx, cov, c, t, nsamples)
    )(jr.split(rng, x.shape[0]), phi, x)
    return -vlb.mean(), s


def elbo_main():
    rng = jax.random.PRNGKey(0)
    N, T = 1, 20
    cov = se_kernel_fn
    noisesd = .5
    logpx = lambda _, s, x: jax.scipy.stats.norm.logpdf(x.reshape(()), jnp.sum(s, 0), noisesd)
    theta_cov = jnp.ones(N)*1.0, jnp.ones(N)*1.0
    theta_tau = jnp.ones(N)*4.0
    theta_x = () # likelihood parameters
    t = jnp.linspace(0, 100, T)

    (s, tau), rng = rngcall(lambda k: \
        vmap(lambda a, b, c: sample_tprocess(a, t, lambda _: _*0, cov, b, c))(
            split(k, N), theta_cov, theta_tau),
        rng)

    phi_tau = jnp.ones(N)*5, jnp.ones(N)*5
    phi_s, rng = rngcall(lambda _: (jnp.zeros((N,T)), jr.normal(_, ((N,T)))*.05), rng)
    theta = theta_x, theta_cov, theta_tau
    phi = phi_s, phi_tau
    nsamples = (5, 10) # (nssamples, nrsamples)
    x, rng = rngcall(lambda k: jax.random.normal(k, (1, T))*noisesd + jnp.sum(s, 0), rng)
    lr = 1e-4
    print(f"ground truth tau: {tau}")
    def step(phi, rng):
        vlb, g = value_and_grad(elbo, 2)(rng, theta, phi, logpx, cov, x, t, nsamples)
        return tree_add(phi, tree_scale(g, lr)), vlb
    nepochs = 100000
    for i in range(1, nepochs):
        rng, key = split(rng)
        phi, vlb = scan(step, phi, split(key, 1000))
        phi_tau_natparams = gamma_natparams_fromstandard(phi[1])
        print(f"{i}: elbo={vlb.mean()}, E[tau]={gamma_mean(phi_tau_natparams)}, V[tau]={gamma_var(phi_tau_natparams)}")


def cvi_main():
    rng = jax.random.PRNGKey(0)
    N, T = 1, 20
    cov = se_kernel_fn
    noisesd = .5
    logpx = lambda _, s, x: jax.scipy.stats.norm.logpdf(x.reshape(()), jnp.sum(s, 0), noisesd)
    theta_cov = jnp.ones(N)*1.0, jnp.ones(N)*1.0
    tau = jnp.ones(N)*4.0
    theta_tau = tau/2-1, -tau/2
    theta_x = () # likelihood parameters
    t = jnp.linspace(0, 100, T)

    (s, tau), rng = rngcall(lambda k: \
        vmap(lambda a, b, c: sample_tprocess(a, t, lambda _: _*0, cov, b, c))(
            split(k, N), theta_cov, tau),
        rng)

    phi_tau = jnp.ones(N)*4-1, -jnp.ones(N)*5

    Kinv = jnp.linalg.inv(compute_K(t, cov, theta_cov).transpose(2,0,1))
    lam = jnp.zeros((N,T)), jnp.zeros((N,T))
    phi_s = lam[0], -.5*Kinv*gamma_meanparams(phi_tau)[1] + vmap(jnp.diag)(lam[1])
    theta = theta_x, theta_cov, theta_tau
    phi = phi_s, phi_tau

    x, rng = rngcall(lambda k: jax.random.normal(k, (1, T))*noisesd + jnp.sum(s, 0), rng)
    lr = 1e-4

    print(f"ground truth tau: {tau}")

    def step(c, rng):
        phi, lam = c    
        phi, lam = cvi_step(rng, theta, phi, lam, logpx, cov, x, t, nsamples=10)[0]
        vlb = cvi_elbo(rng, theta, phi, logpx, cov, x, t, 10)
        return (phi, lam), vlb

    nepochs = 100000
    for i in range(1, nepochs):
        rng, key = split(rng)
        (phi, lam), vlb = scan(step, (phi, lam), split(key, 1000))
        print(f"{i}: E[tau]={gamma_mean(phi[1])}, V[tau]={gamma_var(phi[1])}, {gamma_standardparams(phi[1])}")
        print(vlb.mean())

def cvi_up(theta, phi_s, cov, t):
    N, T = phi_s[0].shape
    theta_cov, theta_tau = theta[1:]
    Ktt = compute_K(t, cov, theta_cov).transpose(2,0,1)
    return tree_add(theta_tau, (.5*T, -.5*jnp.sum(gaussian_meanparams(phi_s)[1]*jnp.linalg.inv(Ktt), (1,2))))

def cvi_down(rng, theta, phi_tau, lam, logpx, cov, x, t, lr, nsamples):
    theta_x, theta_cov, theta_tau = theta
    Kinv = jnp.linalg.inv(compute_K(t, cov, theta_cov).transpose(2,0,1))
    Etau = gamma_meanparams(phi_tau)[1]
    phi_s = lam[0], -.5*Kinv*Etau + vmap(jnp.diag)(lam[1])
    s = gaussian_sample(rng, phi_s, (nsamples,))
    def msg(s, x):
        gf1 = grad(lambda _: logpx(theta_x, _, x))
        gf2 = grad(lambda _: gf1(_).reshape(()))
        g1 = gf1(s)
        g2 = gf2(s)
        return g1 - g2*s, .5*g2
    m = tree_map(partial(jnp.mean, axis=0), vmap(lambda s: vmap(msg, (-1,-1), (-1,-1))(s, x), 0)(s))
    lam = tree_add(tree_scale(lam, 1-lr), tree_scale(m, lr))
    phi_s = lam[0], -.5*Kinv*Etau + vmap(jnp.diag)(lam[1])
    return phi_s, lam, s

def cvi_step(rng, theta, phi, lam, logpx, cov, x, t, nsamples):
    phi_tau = cvi_up(theta, phi[0], cov, t)
    phi_s, lam, s = cvi_down(rng, theta, phi_tau, lam, logpx, cov, x, t, 1e-2, nsamples)
    phi = phi_s, phi_tau
    return (phi, lam), s


def cvi_elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
    T = x.shape[1]
    theta_x, theta_cov, theta_tau = theta
    phi_s, phi_tau = phi
    K = compute_K(t, cov, theta_cov).transpose(2,0,1)
    Kinv = jnp.linalg.inv(K)
    s = gaussian_sample(rng, phi_s, (nsamples,))
    elbo = jnp.mean(jnp.sum(vmap(vmap(logpx, (None,1,1)), (None,0,None))(theta_x,s,x), 1), 0) \
        - jnp.sum(gamma_kl(phi_tau,theta_tau), 0) \
        - .5*jnp.linalg.slogdet(2*jnp.pi*K)[1] \
        + .5*T*gamma_meanparams(phi_tau)[0] \
        - .5*gamma_meanparams(phi_tau)[1]*jnp.sum(Kinv*gaussian_meanparams(phi_s)[1], (1,2)) \
        + gaussian_entropy(phi_s)
    return elbo

if __name__ == "__main__":
    # cvi_main()
    elbo_main()
