import jax
import jax.numpy as jnp
import jax.random as jrd
import pdb

from jax import vmap, lax, random, jit
from jax.tree_util import Partial, tree_map
from functools import partial
from utils import lanczos_tridiag


def tridiag_eig(T):
    def _iter(i, T_i):
        shift_eigs = jnp.linalg.eigh(T_i[-2:, -2:])[0]
        sig = shift_eigs[jnp.argmin(jnp.abs(shift_eigs - A[-1, -1]))]
        c = T_i[0, 0] - sig
        s = T_i[1, 0]
        Qt = jnp.eye(T_i.shape[0]).at[i:i+2, i:i+2].set(
           jnp.array([[c, s],[-s, c]]))
        T_i = Qt@T_i@Qt.T
        return T_i.at[bulge_idx, i].set(0.)
    return T






_identity = lambda _: _
tree_zeros_like = partial(tree_map, jnp.zeros_like)

key = jrd.PRNGKey(8)
Q = jrd.normal(key, (5, 5))
A = Q.T@Q
T = lanczos_tridiag(partial(jnp.dot, A), Q[0] / jnp.linalg.norm(Q[0]),
                    A.shape[0])[0]


pdb.set_trace()
