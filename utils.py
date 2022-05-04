from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js


# inv(L*L.T)*Y
def invcholp(L, Y):
    D = js.linalg.solve_triangular(L, Y, lower=True)
    B = js.linalg.solve_triangular(L.T, D, lower=False)
    return B


# inv(X)*Y
def invmp(X, Y):
    return invcholp(jnp.linalg.cholesky(X), Y)


def gaussian_sample_from_mu_prec(mu, prec, key):
    # reparametrization trick but sampling using precision matrix instead
    L = jnp.linalg.cholesky(prec)
    z = jr.normal(key, mu.shape)
    return mu+js.linalg.solve_triangular(L.T, z, lower=False)
