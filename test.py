import jax
import jax.numpy as jnp
import pdb

from jax import vmap
from time import perf_counter


def f_simple(x):
  A = jnp.arange(10000 * 10).reshape(-1, 10)
  return vmap(lambda _: x[_])(A)


def f_gather(x):
  A = jnp.arange(10000 * 10).reshape(-1, 10)
  return vmap(lambda _: jnp.array([x[(i,)] for i in _]))(A)


def f_slice(x):
  A = jnp.arange(10000 * 10).reshape(-1, 10)
  return vmap(lambda _: jnp.array([x[i] for i in _]))(A)


def f_dynamic_slice(x):
  A = jnp.arange(10000 * 10).reshape(-1, 10)
  return vmap(lambda _: jnp.array([jax.lax.dynamic_index_in_dim(x, i, 0)
                                   for i in _]))(A)




# data
dim = 10000
x = jnp.arange(dim**2).reshape(dim, dim)

# jit + compile
fj_simple = jax.jit(f_simple).lower(x).compile()
fj_gather = jax.jit(f_gather).lower(x).compile()
fj_slice = jax.jit(f_slice).lower(x).compile()
fj_dslice = jax.jit(f_dynamic_slice).lower(x).compile()

# timing
nsamples = 10000
durations = []
for i in range(nsamples):
    tstart = perf_counter()
    fj_simple(x).block_until_ready()
    durations.append((perf_counter()-tstart))
print("Simple: ", jnp.array(durations).mean())


nsamples = 10000
durations = []
for i in range(nsamples):
    tstart = perf_counter()
    fj_gather(x).block_until_ready()
    durations.append((perf_counter()-tstart))
print("Gather: ", jnp.array(durations).mean())


durations = []
for i in range(nsamples):
    tstart = perf_counter()
    fj_slice(x).block_until_ready()
    durations.append((perf_counter()-tstart))
print("Slice: ", jnp.array(durations).mean())


durations = []
for i in range(nsamples):
    tstart = perf_counter()
    fj_dslice(x).block_until_ready()
    durations.append((perf_counter()-tstart))
print("Dynamic: ", jnp.array(durations).mean())




pdb.set_trace()
