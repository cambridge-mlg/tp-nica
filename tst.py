import jax
import jax.numpy as jnp
import jax.random as jrd
from jax import jit, vmap
from jax.tree_util import Partial
from jax.lax import dynamic_slice
from utils import jax_print

import numpy as np
import pdb

N=3
T=5

key = jrd.PRNGKey(0)

A = jrd.normal(key, shape=(N*T, N*T))


@Partial(jit, static_argnames=["N", "T"])
def test(A, N, T):
    return A.reshape(T, N, T, N).swapaxes(1, 2).reshape(-1, N, N)[::(T+1)]

diag_blocks = test(A, N, T)
pdb.set_trace()
