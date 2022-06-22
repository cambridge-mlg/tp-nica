from jax._src.scipy.linalg import lu_factor
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js

import numpy as np
import pdb

from jax.experimental.host_callback import id_tap





# inv(L*L.T)*Y
#def invcholp(L, Y):
#    D = js.linalg.solve_triangular(L, Y, lower=True)
#    B = js.linalg.solve_triangular(L.T, D, lower=False)
#    return B
#
#
## inv(X)*Y
#def invmp(X, Y):
#    return invcholp(jnp.linalg.cholesky(X), Y)


#def gaussian_sample_from_mu_prec(mu, prec, key):
#    # reparametrization trick but sampling using precision matrix instead
#    L = jnp.linalg.cholesky(prec)
#    z = jr.normal(key, mu.shape)
#    return mu+js.linalg.solve_triangular(L.T, z, lower=False)


def rdm_upper_cholesky_of_precision(key, dim):
    P = jr.orthogonal(key, dim)
    key, _ = jr.split(key)
    Q = jnp.diag(1/jr.uniform(key, shape=(dim,)))
    precision_mat = jnp.dot(P.T, jnp.dot(Q, P))
    L = jnp.linalg.cholesky(precision_mat)
    return L.T


def reorder_covmat(cov, N, square=True):
    T_l, T_r = tuple(jnp.int64(_/N) for _ in cov.shape)
    P = jnp.zeros((cov.shape[0], cov.shape[0]))
    for t in range(T_l):
        for n in range(N):
            P = P.at[t*N+n, n*T_l+t].set(1.)
    if square:
        P2 = P
    elif not square:
        P2 = jnp.zeros((cov.shape[1], cov.shape[1]))
        for t in range(T_r):
            for n in range(N):
                P2 = P2.at[t*N+n, n*T_r+t].set(1.)
    return jnp.dot(P, jnp.dot(cov, P2.T))


np.set_printoptions(linewidth=np.inf)
def array_print(arg, transforms):
    print(np.array2string(arg))


def jax_print(x):
    id_tap(tap_func=array_print, arg=x)


def cho_invmp(x, y):
    return js.linalg.cho_solve(js.linalg.cho_factor(x), y)


def cho_inv(x):
    return cho_invmp(x, jnp.eye(x.shape[0]))


def lu_invmp(x, y):
    return js.linalg.lu_solve(js.linalg.lu_factor(x), y)


def lu_inv(x):
    return js.linalg.lu_solve(js.linalg.lu_factor(x), jnp.eye(x.shape[0]))



if __name__=="__main__":
    pdb.set_trace()
