import jax
import jax.numpy as jnp
import pdb

from jax import vmap, lax, random, jit
from jax.tree_util import Partial, tree_map
from functools import partial

_identity = lambda _: _
tree_zeros_like = partial(tree_map, jnp.zeros_like)

@partial(jit, static_argnames=['k'])
def naive_top_k(data, k):
    """Top k implementation built with argmax. Since lax.top_k is slow.
    Adapted from: https://github.com/google/jax/issues/9940
    Faster for smaller k."""

    def top_1(data):
        idx = jnp.argmax(data)
        value = data[idx]
        data = data.at[idx].set(-jnp.inf)
        return data, value, idx

    def scannable_top_1(carry, unused):
        data = carry
        data, value, indice = top_1(data)
        return data, (value, indice)

    data, (values, indices) = lax.scan(scannable_top_1, data, (), k)
    return values.T, indices.T



def _cho_solve(c, b):
    b = b.reshape(c.shape[0], -1)
    b = lax.linalg.triangular_solve(c, b, left_side=True, lower=True,
      transpose_a = False, conjugate_a = False)
    b = lax.linalg.triangular_solve(c, b, left_side=True, lower=True,
                                    transpose_a=True, conjugate_a=True)
    return b.reshape(-1)


@partial(jit, static_argnames=['A_fun', 'nz_max'])
def fsai2(A_fun, G0, num_iter, nz_max, eps, Minv):
    n = G0.shape[0]

    if Minv == None:
        Minv = jnp.eye(n)


    def _G_i_update_fun(k, value):
        i, g_i, idx = value
        phi_grad = jnp.where(jnp.arange(n) < i, 2*A_fun(g_i), 0)
        alpha = -jnp.dot(phi_grad, phi_grad)/jnp.dot(phi_grad, A_fun(phi_grad))
        alpha = lax.cond(jnp.isnan(alpha), tree_zeros_like, _identity, alpha)
        g_i_new = g_i + alpha*phi_grad
        idx = naive_top_k(jnp.abs(g_i_new), nz_max)[1]
        g_i = jnp.zeros_like(g_i_new).at[idx].set(g_i_new[idx])
        g_i = g_i.at[i].set(g_i_new[i])
        return (i, g_i, idx)


    def _calc_G_i(i, G0_i):
        idx = naive_top_k(jnp.abs(G0_i), nz_max)[1]
        init_val = (i, G0_i, idx)
        i, Gk_i, idx = lax.fori_loop(0, num_iter, _G_i_update_fun, init_val)
        d_ii = jnp.dot(Gk_i, A_fun(Gk_i))**-0.5
        return d_ii*Gk_i


    G0 = lax.cond(jnp.all(G0 == jnp.eye(n)), _identity,
              lambda _: (1/jnp.einsum('ii->i', _))[:, None] * _, G0)
    G = vmap(_calc_G_i, (0, 0))(jnp.arange(n), G0)
    return G


key = random.PRNGKey(8)
T = 1024
N = 6

@jit
def _fun(k):
    Q = random.normal(k, (T, N, N))
    A = vmap(lambda x: x.T@x)(Q)
    P = fsai2(lambda b: _cho_solve(A, b), jnp.eye(T*N), 2, 10, 1e-8, None)
    return P

Ps = vmap(_fun)(random.split(key, 100))

pdb.set_trace()
