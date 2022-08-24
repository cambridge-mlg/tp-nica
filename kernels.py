from jax._src.tree_util import tree_map
import jax.numpy as jnp
import jax.random as jr

import pdb

from jax import vmap
from jax import lax
from jax.tree_util import Partial


def squared_euclid_dist(x, y):
    return jnp.sum((x-y)**2, -1)


def squared_euclid_dist_mat(x):
    return vmap(lambda _x: vmap(lambda _y: squared_euclid_dist(_x, _y))(x))(x)


def se_kernel_fn(x, y, params):
    sigma, lscale = params
    k = sigma**2 * jnp.exp(-0.5*squared_euclid_dist(x, y) / lscale**2)
    return k+(x==y).squeeze()*1e-5


def bound_se_kernel_params(params, sigma_min=1e-3, ls_min=1, ls_max=900):
    sigma, lscale = params
    sigma = tree_map(lambda _: _+sigma_min, sigma)
    lscale = tree_map(lambda _: ls_max/(_ + 1.) + ls_min, lscale)
    return (sigma, lscale)


def compute_K(x, kernel_fn, params):
    k_fun = Partial(kernel_fn, params=params)
    return vmap(lambda _x: vmap(lambda _y: k_fun(_x, _y))(x))(x)


def rdm_SE_kernel_params(key, min_lscale=25., max_lscale=100.,
                         sd_min=0.1, sd_max=1.):
    """
    Note: x is needed to find reasonable length-scale thats not too smooth
    """
    lscale = jr.uniform(key, minval=min_lscale,
                        maxval=max_lscale)
    key, sd_key = jr.split(key)
    sd = jr.uniform(sd_key, minval=sd_min, maxval=sd_max)
    return (sd, lscale)


def rdm_df(key, maxval=4):
    """
    Note: df > 2 has been hard-coded
    Note: forcing alpha=beta=nu/2
    Note: max_val is exclusive
    """
    return jr.uniform(key, minval=2, maxval=maxval)
