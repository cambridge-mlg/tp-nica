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
    sigma, lscale, noise = params
    k = sigma**2 * jnp.exp(-0.5*squared_euclid_dist(x, y) / lscale**2)
    return k+(x==y).squeeze()*noise**2


def compute_K(x, kernel_fn, params):
    k_fun = Partial(kernel_fn, params=params)
    return vmap(lambda _x: vmap(lambda _y: k_fun(_x, _y))(x))(x)


def rdm_SE_kernel_params(key, x, lscale_min_multip=15., lscale_max_multip=50.,
                         sd_min=0.1, sd_max=1., noise_multip=1e-6):
    """
    Note: x is needed to find reasonable length-scale thats not too smooth
    """
    D = jnp.sqrt(squared_euclid_dist_mat(x))
    min_val = jnp.min(D[jnp.tril_indices(D.shape[0], k=-1)])
    lscale = jr.uniform(key, minval=lscale_min_multip*min_val,
                        maxval=lscale_max_multip*min_val)
    key, sd_key = jr.split(key)
    sd = jr.uniform(sd_key, minval=sd_min, maxval=sd_max)
    noise_sd = noise_multip * jr.uniform(key, minval=sd_min, maxval=sd_max)
    return (sd, lscale, noise_sd)


def rdm_df(key, maxval=4):
    """
    Note: df > 2 has been hard-coded
    Note: forcing alpha=beta=nu/2
    Note: max_val is exclusive
    """
    return jr.uniform(key, minval=2, maxval=maxval)
