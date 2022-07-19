import jax
import jax.random as jrd
import jax.numpy as jnp

import pdb
import matplotlib.pyplot as plt

from jax import vmap
from util import rngcall

rng = jrd.PRNGKey(0)
T = 200
N = 3

mu_g = jnp.load("mu_g.npy")[0]
mu_b = jnp.load("mu_b.npy")[0]
mu_t = jnp.load("mu_t.npy")[0]

cov_g = jnp.load("cov_g.npy")[0]
cov_b = jnp.load("cov_b.npy")[0]
cov_t = jnp.load("cov_t.npy")[0]

s, rng = rngcall(lambda _: vmap(lambda k, a, b: jrd.multivariate_normal(k, a, b))(
                jrd.split(_, cov_g.shape[0]), mu_t.reshape(-1, T, N), cov_t), rng)




plt.plot(s[:, :, 0].T)
plt.show()
plt.plot(s[:, :, 1].T)
plt.show()
plt.plot(s[:, :, 2].T)
plt.show()

