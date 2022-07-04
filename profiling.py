from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import optax

import pdb

from jax import vmap, jit, value_and_grad
from jax.lax import cond
from jax.tree_util import tree_map
from util import jax_profiler, rngcall, mvp, tree_get_idx
from utils import rdm_upper_cholesky_of_precision, jax_print, reorder_covmat
from utils import cho_inv, cho_invmp, lu_inv, lu_invmp, comp_K_N
from tprocess.kernels import compute_K, se_kernel_fn, rdm_SE_kernel_params
from tprocess.sampling import gen_1d_locations

key = jr.PRNGKey(0)
N = 3
n_pseudo = 5
T = 10

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


Kuu = vmap(lambda b: vmap(lambda a:
    comp_K_N(a, b, se_kernel_fn, theta_k)+1e-6*jnp.eye(N)
)(tu))(tu)
Kuu = Kuu.swapaxes(1, 2).reshape(n_pseudo*N, n_pseudo*N)
Ksu = vmap(lambda b:vmap(lambda a:
    comp_K_N(a, b, se_kernel_fn, theta_k))(tu))(t)
Ksu = Ksu.swapaxes(1, 2).reshape(T*N, n_pseudo*N)
kss = vmap(
    lambda tc: vmap(lambda t: se_kernel_fn(t, t, tc))(t)
    )(theta_k)


WTy = jnp.einsum('ijk,ik->jk', W, y).T.reshape(-1, 1)
L = js.linalg.block_diag(*jnp.moveaxis(
  jnp.einsum('ijk, ilk->jlk', W, W), -1, 0))
Kyy_inv = jnp.eye(L.shape[0])+L@Kuu
lu_fact = js.linalg.lu_factor(Kyy_inv)

solved = js.linalg.lu_solve(lu_fact, WTy)


P = jnp.zeros_like(Kuu)
for i in range(n_pseudo):
    for j in range(N):
        P = P.at[:,:].add(jnp.outer(jnp.kron(jnp.eye(n_pseudo)[i], jnp.eye(N)[j]),
                       jnp.kron(jnp.eye(N)[j], jnp.eye(n_pseudo)[i])))

mu_s = Ksu @ js.linalg.lu_solve(lu_fact, WTy)
cov_s = vmap(lambda X, y: jnp.diag(y)-X@js.linalg.lu_solve(lu_fact, L)@X.T,
          in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss)
s, key = rngcall(lambda _: jr.multivariate_normal(_, mu_s.reshape(T, N),
                                                  cov_s, shape=(2, T)), key)

# profiling
theta = (Kuu, P, n_pseudo, N)


def v8(key, theta, s):
    Kuu_reord, P, np, N = theta
    _ = key
    s = s + 1e-16
    return s, jnp.linalg.inv(Kuu_reord)+s


def v9(key, theta, s):
    Kuu_reord, P, np, N = theta
    _ = key
    s = s + 1e-16
    return s, P@jnp.linalg.inv(P.T@Kuu_reord@P)@P.T + s


def v10(key, theta, s):
    Kuu_reord, P, np, N = theta
    _ = key
    s = s + 1e-16
    block_diag = P.T@Kuu_reord@P
    blocks_inv = js.linalg.block_diag(*[jnp.linalg.inv(block_diag[i*np:(i+1)*np,
                                        i*np:(i+1)*np]) for i in range(N)])
    return s, P@Kuu_reord@P.T + s


v8_time = jax_profiler(theta, jnp.array(1e-16), v8, 1e4, 3)
v9_time = jax_profiler(theta, jnp.array(1e-16), v9, 1e4, 3)
v10_time = jax_profiler(theta, jnp.array(1e-16), v10, 1e4, 3)


pdb.set_trace()
