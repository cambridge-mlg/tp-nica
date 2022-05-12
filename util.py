import operator
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

# batch utils
vdot = lambda x, y: jnp.sum(x*y, -1)
transpose = lambda _: jnp.swapaxes(_, -1, -2)
mvp = lambda X, v: jnp.matmul(X, v[...,None]).squeeze(-1)
outer = lambda x, y: x[...,None]*y[...,None,:]

# call rng function and return value with new rng
def rngcall(f, rng, *args, **kwargs):
    rng1, rng2 = jax.random.split(rng)
    return f(rng1, *args, **kwargs), rng2


def tree_add(tree1, tree2):
    return tree_map(operator.add, tree1, tree2)


def tree_sub(tree1, tree2):
    return tree_map(operator.sub, tree1, tree2)


def tree_mul(tree1, tree2):
    return tree_map(operator.mul, tree1, tree2)


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
