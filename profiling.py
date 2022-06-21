from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import optax

import pdb

from jax import vmap, jit, value_and_grad
from jax.tree_util import tree_map
from util import profile, rngcall, mvp
from utils import rdm_upper_cholesky_of_precision, jax_print, reorder_covmat
from utils import cho_inv
from tprocess.kernels import compute_K, se_kernel_fn, rdm_SE_kernel_params
from tprocess.sampling import gen_1d_locations

key = jr.PRNGKey(0)
N = 3
n_pseudo = 5
T = 10

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
#    s = s + 1e-16
#    return s, mvp(WhatT, yhat.T).T+s
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
#    s = s + 1e-16
#    return s, out+s
#
#
#v2_out = v2(key, theta, jnp.zeros(1))[1]
#
#
#def v3(key, theta, s):
#    What, yhat = theta
#    _ = key
#    out = jnp.einsum('ijk,ik->jk', What, yhat)
#    s = s + 1e-16
#    return s, out+s
#
#v3_out = v3(key, theta, jnp.zeros(1))[1]
#
#
#v1_time = profile(theta, jnp.zeros(1), v1, 1e5, 4)
#v2_time = profile(theta, jnp.zeros(1), v2, 1e5, 4)
#v3_time = profile(theta, jnp.zeros(1), v3, 1e5, 4)
#print(v1_time, v2_time, v3_time)


# profiling matrix inversion
t = gen_1d_locations(T)
tu, key = rngcall(lambda k: jr.uniform(k, shape=(n_pseudo, 1),
                                       minval=0, maxval=jnp.max(t)), key)
theta_k, key = rngcall(
   lambda _k: vmap(lambda _: rdm_SE_kernel_params(_, t)
                  )(jr.split(_k, N)), key)
Kuu = compute_K(tu, se_kernel_fn, theta_k).transpose(2, 0, 1) + 1e-6*jnp.eye(len(tu))
Kuu_full = js.linalg.block_diag(*Kuu)
Kuu_reord = reorder_covmat(Kuu_full, N)
theta = Kuu_reord


def v4(key, theta, s):
    K = theta
    _ = key
    s = s + 1e-16
    return s, jnp.linalg.inv(K)+s


v4_time = profile(theta, jnp.ones(1), v4, 1e4, 3)


def v5(key, theta, s):
    K = theta
    _ = key
    s = s + 1e-16
    return s, cho_inv(K)+s


v5_time = profile(theta, jnp.ones(1), v5, 1e4, 3)

print("linalg.inv: ", v4_time, "cho_inv: ", v5_time)

pdb.set_trace()
