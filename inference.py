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
from tprocess.kernels import se_kernel_fn
from tprocess.sampling import sample_tprocess
from util import *
from gamma import *
from gaussian import *

def elbo_s_natparams(rng, theta, phi, logpx, cov, x, t, rinv, nsamples):
    theta_x, theta_cov, theta_r = theta
    (What, yhat), phi_r = phi
    scale = (2*(theta_r[0]+1)-2)/(-2*rinv*theta_r[1])
    Ktt = vmap(lambda tc: \
            vmap(lambda t1:
                vmap(lambda t2:
                    cov(t1, t2, tc)
                )(t)
            )(t)
        )(theta_cov)*scale[:,None,None]
    qs_r = (What*yhat), -.5*(jnp.linalg.inv(Ktt) + vmap(jnp.diag)(jnp.square(What)))
    mu_s, Vs = gaussian_standardparams(qs_r)
    s, rng = rngcall(jax.random.multivariate_normal, rng, mu_s, Vs, (nsamples,mu_s.shape[0]))
    elbo = jnp.sum(gaussian_logZ2(qs_r), 0) \
        - jnp.sum(mu_s*What*yhat) \
        - jnp.sum(-.5*jnp.square(What)*(vmap(jnp.diag)(Vs) + jnp.square(mu_s))) \
        + jnp.mean(jnp.sum(vmap(vmap(logpx, (None,1,1)), (None,0,None))(theta_x,s,x), 1), 0)
    return elbo

# compute estimate of elbo terms that depend on rinv
def elbo_s(rng, theta, phi, logpx, cov, x, t, rinv, nsamples):
    theta_x, theta_cov, theta_r = theta
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
    # x: (M,T)
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
    theta_x, theta_cov, theta_r = theta
    phi_s, phi_r = phi
    nu, rho = phi_r
    pdb.set_trace()
    rinv, rng = rngcall(lambda _: jax.random.gamma(_, nu/2, (nsamples_r, *phi_r[0].shape))/(rho/2), rng)
    kl = jnp.sum(gamma_kl(gamma_natparams_fromweird(phi_r), gamma_natparams_fromweird(theta_r)), 0)
    vlb_r, s = vmap(lambda _: elbo_s(rng, theta, phi, logpx, cov, x, t, _, nsamples_s))(rinv)
    return jnp.mean(vlb_r, 0) - kl, None

# compute elbo estimate, assumes q(r) is gamma
def elbo_natparams(rng, theta, phi, logpx, cov, x, t, nsamples):
    nsamples_s, nsamples_r = nsamples
    theta_x, theta_cov, theta_r = theta
    phi_s, phi_r = phi
    nu, rho = phi_r
    rinv, rng = rngcall(lambda _: jax.random.gamma(_, phi_r[0]+1, (nsamples_r, *phi_r[0].shape))/(-phi_r[1]), rng)
    kl = jnp.sum(gamma_kl(phi_r, theta_r), 0)
    vlb_r = vmap(lambda _: elbo_s_natparams(rng, theta, phi, logpx, cov, x, t, _, nsamples_s))(rinv)
    return jnp.mean(vlb_r, 0) - kl

# compute elbo estimate, assumes q(r) is gamma
def elbo_grads(rng, theta, phi, logpx, cov, x, t, nsamples):
    nsamples_s, nsamples_r = nsamples
    theta_x, theta_cov, theta_r = theta
    phi_s, phi_r = phi
    rinv, rng = rngcall(lambda _: jax.random.gamma(_, phi_r[0]+1, (nsamples_r, *phi_r[0].shape))/(-phi_r[1]), rng)
    kl = jnp.sum(gamma_kl(phi_r, theta_r), 0)
    def foo(p):
        vlb = vmap(lambda _: elbo_s_natparams(rng, theta, (p, phi_r), logpx, cov, x, t, _, nsamples_s))(rinv)
        return jnp.mean(vlb, 0), vlb
    g_s, vlb_r = grad(foo, has_aux=True)(phi_s)
    g_r = tree_sub(
        tree_map(partial(jnp.mean, axis=0), tree_mul(tree_sub(gamma_stats(rinv), gamma_meanparams(phi_r)), (vlb_r[:,None], vlb_r[:,None]))), \
        grad(lambda _: jnp.sum(gamma_kl(_, theta_r), 0))(phi_r))
    return g_s, g_r

@jit
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
    N, T = 5, 10
    cov = se_kernel_fn
    noisesd = .2
    logpx = lambda _, s, x: jax.scipy.stats.norm.logpdf(x.reshape(()), jnp.sum(s, 0), noisesd)
    tau = jnp.ones(N)*4.0
    theta_cov = jnp.ones(N)*1.0, jnp.ones(N)*1.0
    theta_r = tau/2, tau/2
    theta_x = () # likelihood parameters
    phi_r = jnp.ones(N)*10, jnp.ones(N)
    phi_s, rng = rngcall(lambda _: (jnp.zeros((N,T)), jr.normal(_, ((N,T)))), rng)
    theta = theta_x, theta_cov, theta_r
    phi = phi_s, phi_r
    nsamples = (5, 10) # (nssamples, nrsamples)
    t = jnp.linspace(0, 100, T)
    (s, rinv), rng = rngcall(lambda k: \
        vmap(lambda a, b, c: sample_tprocess(a, t, lambda _: _*0, cov, b, c))(
            split(k, N), theta_cov, (nu, rho)),
        rng)
    x, rng = rngcall(lambda k: jax.random.normal(k, (1, T))*noisesd + jnp.sum(s, 0), rng)
    lr = 1e-4
    print(f"ground truth rinv: {rinv}")
    def step(phi, rng):
        # vlb, g = value_and_grad(elbo, 2, has_aux=True)(rng, theta, phi, logpx, cov, x, t, nsamples)
        vlb, g = value_and_grad(elbo_natparams, 2)(rng, theta, phi, logpx, cov, x, t, nsamples)
        # g = elbo_grads(rng, theta, phi, logpx, cov, x, t, nsamples)
        # vlb = elbo_natparams(rng, theta, phi, logpx, cov, x, t, nsamples)
        return tree_add(phi, tree_scale(g, lr)), vlb
    nepochs = 100000
    print(s[0])
    for i in range(1, nepochs):
        rng, key = split(rng)
        phi, vlb = scan(step, phi, split(key, 1000))
        # print(f"{i}: {jnp.corrcoef(s_samples.mean(axis=(0,1,2,3)), s[0])[0,1]}, E[rinv]={gamma_mean(phi_natparams)}, V[rinv]={gamma_var(phi_natparams)}")
        print(f"{i}: elbo={vlb.mean()}, E[rinv]={gamma_mean(phi[1])}, V[rinv]={gamma_var(phi[1])}")
    # print(s_samples.mean(axis=(0,1,2,3)))
    # print(s[0])


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
