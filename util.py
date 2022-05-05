import operator
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_multimap

# batch utils
vdot = lambda x, y: jnp.sum(x*y, -1)
transpose = lambda _: jnp.swapaxes(_, -1, -2)


# call rng function and return value with new rng
def rngcall(f, rng, *args, **kwargs):
    rng1, rng2 = jax.random.split(rng)
    return f(rng1, *args, **kwargs), rng2


def tree_add(tree1, tree2):
    return tree_multimap(operator.add, tree1, tree2)


def tree_sub(tree1, tree2):
    return tree_multimap(operator.sub, tree1, tree2)


def tree_mul(tree1, tree2):
    return tree_multimap(operator.mul, tree1, tree2)


def tree_scale(tree, c):
    return tree_map(lambda _: c*_, tree)


# TP/GP mean functions
def zero_mean_fn(x):
    return jnp.zeros(x.shape[0])


def cos_1d_mean_fn(x):
    return jnp.cos(x)
