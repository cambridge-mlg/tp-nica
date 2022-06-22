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
from utils import cho_inv, cho_invmp, lu_inv, lu_invmp
from tprocess.kernels import compute_K, se_kernel_fn, rdm_SE_kernel_params
from tprocess.sampling import gen_1d_locations

key = jr.PRNGKey(0)
N = 4
n_pseudo = 50
T = 100

W, key = rngcall(lambda _k: vmap(
    lambda _: rdm_upper_cholesky_of_precision(_, N), out_axes=-1)(
        jr.split(_k, n_pseudo)), key
)
y = jnp.ones(shape=(N, n_pseudo))
theta = W, y

# profiling matrix inversion
t = gen_1d_locations(T)
tu, key = rngcall(lambda k: jr.uniform(k, shape=(n_pseudo, 1),
                                       minval=0, maxval=jnp.max(t)), key)
theta_k, key = rngcall(
   lambda _k: vmap(lambda _: rdm_SE_kernel_params(_, t)
                  )(jr.split(_k, N)), key)
Kuu = compute_K(tu, se_kernel_fn, theta_k).transpose(2, 0, 1) + 1e-6*jnp.eye(len(tu))
Ksu = vmap(lambda tc:
        vmap(lambda t1:
            vmap(lambda t2: se_kernel_fn(t1, t2, tc))(tu)
        )(t)
    )(theta_k)

Kuu_full = js.linalg.block_diag(*Kuu)
Ksu_full = js.linalg.block_diag(*Ksu)
Kuu_reord = reorder_covmat(Kuu_full, N)
Ksu_reord = reorder_covmat(Ksu_full, N, square=False)

WTy = jnp.einsum('ijk,ik->jk', W, y).T.reshape(-1, 1)
L = js.linalg.block_diag(*jnp.moveaxis(
  jnp.einsum('ijk, ilk->jlk', W, W), -1, 0))

# profiling
theta = (Kuu_reord, L, WTy)


def v6(key, theta, s):
    Kuu, L, WTy = theta
    _ = key
    s = s + 1e-16
    Kuu_inv = cho_inv(Kuu)
    return s, Kuu_inv @ cho_invmp(Kuu_inv+L, WTy)


def v7(key, theta, s):
    Kuu, L, WTy = theta
    _ = key
    s = s + 1e-16
    Kuu_inv = jnp.linalg.inv(Kuu)
    return s, Kuu_inv @ jnp.linalg.inv(Kuu_inv+L) @ WTy


def v8(key, theta, s):
    Kuu, L, WTy = theta
    _ = key
    s = s + 1e-16
    return s, lu_invmp(jnp.eye(L.shape[0])+L@Kuu, WTy)


v5_time = profile(theta, jnp.ones(1), v6, 1e4, 3)
v6_time = profile(theta, jnp.ones(1), v7, 1e4, 3)
v7_time = profile(theta, jnp.ones(1), v8, 1e4, 3)

print(v5_time, v6_time, v7_time)
pdb.set_trace()
