from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import optax
import timeit

import pdb

from jax import vmap, jit, value_and_grad

from utils import sample_wishart


key = jr.PRNGKey(0)
N = 12
M = 12
T = 20000

# prof settings

if __name__=="__main__":

    W = vmap(sample_wishart, in_axes=(0, None, None))(jr.split(key, T),
                                                      N+1., jnp.eye(N))

    def f0(x):
        return vmap(jnp.matmul)(x, x.swapaxes(1, 2))


    def f1(x):
        return jnp.matmul(x, x.swapaxes(1, 2))


    f0_comp = jit(f0).lower(W).compile()
    f1_comp = jit(f1).lower(W).compile()

    print(timeit.repeat("f0_comp(W).block_until_ready()",
                        "from __main__ import f0_comp, W",
                        number=100000))
    print(timeit.repeat("f1_comp(W).block_until_ready()",
                        "from __main__ import f1_comp, W",
                        number=100000))

    pdb.set_trace()










