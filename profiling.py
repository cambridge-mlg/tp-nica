from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax

import pdb

from jax import vmap, jit, value_and_grad
from jax.tree_util import tree_map
from util import profile, rngcall, mvp
from utils import rdm_upper_cholesky_of_precision, jax_print

key = jr.PRNGKey(0)
N = 10
n_pseudo = 1000
T = 10000

#W, key = rngcall(lambda _k: vmap(
#    lambda _: rdm_upper_cholesky_of_precision(_, N), out_axes=-1)(
#        jr.split(_k, n_pseudo)), key
#)
#y = jnp.ones(shape=(N, n_pseudo))
#theta = W, y
#
#
#def v1(key, theta, s):
#    What, yhat = theta
#    _ = key
#    WhatT = jnp.swapaxes(What, -1, 0)
#    return s, mvp(WhatT, yhat.T).T
#
#
#v1_out = v1(key, theta, jnp.zeros(1))[1]
#
#
#def v2(key, theta, s):
#    What, yhat = theta
#    _ = key
#    out = vmap(lambda _W, _y: mvp(_W.T, _y),
#               in_axes=(-1, -1), out_axes=-1)(What, yhat)
#    return s, out
#
#
#v2_out = v2(key, theta, jnp.zeros(1))[1]
#
#
#def v3(key, theta, s):
#    What, yhat = theta
#    _ = key
#    out = jnp.einsum('ijk,ik->jk', What, yhat)
#    return s, out
#
#v3_out = v3(key, theta, jnp.zeros(1))[1]
#
#
#v1_time = profile(theta, jnp.zeros(1), v1, 1e5, 4)
#v2_time = profile(theta, jnp.zeros(1), v2, 1e5, 4)
#v3_time = profile(theta, jnp.zeros(1), v3, 1e5, 4)
#print(v1_time, v2_time, v3_time)

tu, key = rngcall(lambda k: jr.uniform(k, shape=(n_pseudo, 1),
                                       minval=0, maxval=jnp.max(T)), key)







pdb.set_trace()
