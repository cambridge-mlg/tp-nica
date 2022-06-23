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

from functools import partial
from math import factorial
from tprocess.kernels import se_kernel_fn, compute_K
from tprocess.sampling import sample_tprocess
from utils import reorder_covmat, jax_print
from util import *
from gamma import *
from gaussian import *


def structured_elbo_s(rng, theta, phi_s, logpx, cov_fn, x, t, tau, nsamples):
    theta_x, theta_cov = theta[:2]
    What, yhat, tu = phi_s
    N, n_pseudo = yhat.shape
    T = t.shape[0]
    Kuu = compute_K(tu, cov_fn, theta_cov).transpose(2, 0, 1)/tau[:, None, None]\
        + 1e-6*jnp.eye(len(tu))
    Ksu = vmap(lambda tc:
            vmap(lambda t1:
                vmap(lambda t2: cov_fn(t1, t2, tc))(tu)
            )(t)
        )(theta_cov)/tau[:, None, None]
    kss = vmap(
        lambda tc: vmap(lambda t: cov_fn(t, t, tc))(t)
        )(theta_cov) / tau[:, None]
    # construct large block diagonal cov matrices
    Kuu_full = js.linalg.block_diag(*Kuu)
    Ksu_full = js.linalg.block_diag(*Ksu)
    # re-order the matrices to have same time points next to each other
    Kuu_reord = reorder_covmat(Kuu_full, N)
    Ksu_reord = reorder_covmat(Ksu_full, N, square=False)

    # compute parameters for \tilde{q(s|tau)}
    WTy = jnp.einsum('ijk,ik->jk', What, yhat).T.reshape(-1, 1)
    L = js.linalg.block_diag(*jnp.moveaxis(
      jnp.einsum('ijk, ilk->jlk', What, What), -1, 0))
    LK = L@Kuu_reord
    lu_fact = js.linalg.lu_factor(jnp.eye(L.shape[0])+LK)
    KyyWTy = js.linalg.lu_solve(lu_fact, WTy)
    mu_s = Ksu_reord @ KyyWTy
    cov_s = vmap(lambda X, y: jnp.diag(y)-X@js.linalg.lu_solve(lu_fact, L)@X.T,
          in_axes=(0, -1))(Ksu_reord.reshape(T, N, -1), kss)
    s, rng = rngcall(lambda _: jr.multivariate_normal(_, mu_s.reshape(T, N),
        cov_s, shape=(nsamples, T)), rng)

    # compute E_{\tilde{q(s|tau)}}[log_p(x_t|s_t)]
    Elogpx = jnp.mean(
        jnp.sum(vmap(lambda _: vmap(logpx,(1, 0, None))(x, _, theta_x))(s), 1)
    )

    # compute KL[q(u)|p(u)]
    tr = jnp.trace(js.linalg.lu_solve(lu_fact, LK.T, trans=1).T)
    h = Kuu_reord@KyyWTy
    logZ = -0.5*(-jnp.dot(WTy.squeeze(), h)
                 +jnp.linalg.slogdet(jnp.eye(L.shape[0])+LK)[1])
    KLqpu = -0.5*(tr+h.T@L@h)+WTy.T@h - logZ
    return Elogpx-KLqpu, s


# compute elbo estimate, assumes q(r) is gamma
def structured_elbo(rng, theta, phi, logpx, cov_fn, x, t, nsamples):
    nsamples_s, nsamples_tau = nsamples
    theta_tau = theta[2]
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


def meanfield_elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
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


def cvi_up(theta, phi_s, cov, t):
    N, T = phi_s[0].shape
    theta_cov, theta_tau = theta[1:]
    Ktt = compute_K(t, cov, theta_cov).transpose(2,0,1)
    return tree_add(theta_tau, (.5*T, -.5*jnp.sum(gaussian_meanparams(phi_s)[1]*jnp.linalg.inv(Ktt), (1,2))))


def cvi_down(rng, theta, phi_tau, lam, logpx, cov_fn, x, t, lr, nsamples):
    theta_x, theta_cov, theta_tau = theta
    Kinv = jnp.linalg.inv(compute_K(t, cov_fn, theta_cov).transpose(2,0,1))
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
        vmap(lambda a, b, c: sample_tprocess(a, t, lambda _: .0, cov, b, c))(
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
        vlb = meanfield_elbo(rng, theta, phi, logpx, cov, x, t, 10)
        return (phi, lam), vlb

    nepochs = 100000
    for i in range(1, nepochs):
        rng, key = split(rng)
        (phi, lam), vlb = scan(step, (phi, lam), split(key, 1000))
        print(f"{i}: elbo={vlb.mean()}, E[tau]={gamma_mean(phi[1])}, V[tau]={gamma_var(phi[1])}")


if __name__ == "__main__":
    # cvi_main()
    elbo_main()
