import operator
import jax
import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jax.random import split
from jax.tree_util import tree_map

from functools import partial
from time import perf_counter

# batch utils
vdot = lambda x, y: jnp.sum(x*y, -1)
transpose = lambda _: jnp.swapaxes(_, -1, -2)
mvp = lambda X, v: jnp.matmul(X, v[...,None]).squeeze(-1)
mmp = lambda X, Y: jnp.matmul(X, Y)
outer = lambda x, y: x[...,None]*y[...,None,:]

# call rng function and return value with new rng
def rngcall(f, rng, *args, **kwargs):
    rng1, rng2 = jax.random.split(rng)
    return f(rng1, *args, **kwargs), rng2


tree_add = partial(tree_map, operator.add)
tree_sub = partial(tree_map, operator.sub)
tree_mul = partial(tree_map, operator.mul)


def tree_scale(tree, c):
    return tree_map(lambda _: c*_, tree)


def tree_get_idx(tree, idx):
    """Get idx row from each leaf of pytree"""
    return tree_map(lambda a: a[idx], tree)


def tree_get_range(tree, start_idx, stop_idx):
    """Get range of rows from each leaf of pytree"""
    return tree_map(lambda a: a[start_idx:stop_idx], tree)


# TP/GP mean functions
def zero_mean_fn(x):
    return jnp.zeros(x.shape[0])


def cos_1d_mean_fn(x):
    return jnp.cos(x)


def jax_profiler(theta, state, step, nsteps, nsamples):
    rng = jax.random.PRNGKey(0)
    durations = []
    f = jax.jit(lambda k: scan(lambda s, k: step(k, theta, s), state,
                               split(k, nsteps))[1]).lower(rng).compile()
    for i in range(nsamples):
        tstart = perf_counter()
        f(rng).block_until_ready()
        durations.append((perf_counter()-tstart))
    return jnp.median(jnp.array(durations)).item()/nsteps
