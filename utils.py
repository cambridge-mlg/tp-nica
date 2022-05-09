from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js

import scipy as sp
import pdb

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


def rdm_upper_cholesky_of_precision(key, dim):
    P = jr.orthogonal(key, dim)
    key, _ = jr.split(key)
    Q = jnp.linalg.inv(jnp.diag(jr.uniform(key, shape=(dim,))))
    precision_mat = jnp.dot(P.T, jnp.dot(Q, P))
    L = jnp.linalg.cholesky(precision_mat)
    return L.T


if __name__=="__main__":
    pdb.set_trace()
