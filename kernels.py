from jax._src.tree_util import tree_map
import jax.numpy as jnp
import jax.random as jr

import pdb

from jax import vmap, jit
from jax import lax
from jax.tree_util import Partial
from utils import jax_print, _identity


@jit
def squared_euclid_dist(x, y):
    return jnp.sum((x-y)**2, -1)


@jit
def squared_euclid_dist_mat(x):
    return vmap(vmap(squared_euclid_dist, (None, 0)), (0, None))(x, x)


@jit
def se_kernel_fn(x, y, params, jitter=1e-5):
    sigma, lscale = params
    k = sigma**2 * jnp.exp(-0.5*squared_euclid_dist(x, y) / lscale**2)
    return lax.cond(jnp.all(x == y), lambda _: _ + jitter, _identity, k)


def bound_se_kernel_params(params, sigma_min=1e-3, ls_min=1, ls_max=900):
    sigma, lscale = params
    sigma = tree_map(lambda _: _+sigma_min, sigma)
    #lscale = tree_map(lambda _: ls_max/(_ + 1.) + ls_min, lscale)
    lscale = tree_map(lambda _: jnp.clip(_, ls_min, ls_max), lscale)
    return (sigma, lscale)


def compute_K(x, kernel_fn, params):
    return vmap(vmap(kernel_fn, (None, 0, None)), (0, None, None))(x, x, params)


def rdm_SE_kernel_params(key, min_lscale=0.1, max_lscale=0.5,
                         sd_min=0.1, sd_max=1.):
    """
    Note: x is needed to find reasonable length-scale thats not too smooth
    """
    lscale = jr.uniform(key, minval=min_lscale,
                        maxval=max_lscale)
    key, sd_key = jr.split(key)
    sd = jr.uniform(sd_key, minval=sd_min, maxval=sd_max)
    return (sd, lscale)


def rdm_df(key, min_val, max_val):
    """
    Note: df > 2 needed
    """
    return jr.uniform(key, minval=min_val, maxval=max_val)
