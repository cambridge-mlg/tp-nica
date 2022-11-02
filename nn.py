import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy as jsp
import jax.nn as nn

from jax import vmap, jit, lax
from utils import jax_print

import pdb


def l2normalize(W, axis=0):
    """Normalizes MLP weight matrices.
    Args:
        W (matrix): weight matrix.
        axis (int): axis over which to normalize.
    Returns:
        Matrix l2 normalized over desired axis.
    """
    l2norm = jnp.sqrt(jnp.sum(W*W, axis, keepdims=True))
    W = W / l2norm
    return W


def SmoothLeakyRelu(slope):
    def smooth_leaky_relu(x, alpha=slope):
        """Calculate smooth leaky ReLU on an input.
        Source: https://stats.stackexchange.com/questions/329776/ \
                approximating-leaky-relu-with-a-differentiable-function
        Args:
            x (float): input value.
            alpha (float): controls level of nonlinearity via slope.
        Returns:
            Value transformed by the smooth leaky ReLU.
        """
        return alpha*x + (1 - alpha)*jnp.logaddexp(x, 0)
    return smooth_leaky_relu


def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x


def unif_nica_layer(N, M, key, iter_4_cond=1e4):
    def _gen_matrix(N, M, key):
        A = jrandom.uniform(key, (N, M), minval=-2., maxval=2.)
        #A = l2normalize(A)
        _cond = jnp.linalg.cond(A)
        return A, _cond

    # generate multiple matrices
    keys = jrandom.split(key, iter_4_cond)
    A, conds = vmap(_gen_matrix, (None, None, 0))(N, M, keys)
    target_cond = jnp.percentile(conds, 25)
    target_idx = jnp.argmin(jnp.abs(conds-target_cond))
    return A[target_idx]


def init_nica_params(key, N, M, nonlin_layers, repeat_layers):
    '''BEWARE: Assumes factorized distribution
        and equal width in all hidden layers'''
    layer_sizes = [N] + [M]*nonlin_layers + [M]
    keys = jrandom.split(key, len(layer_sizes)-1)
    if repeat_layers:
        _keys = keys
        keys = jnp.repeat(_keys[0][None], _keys.shape[0], 0)
        if N != M:
            keys = keys.at[1:].set(_keys[-1])
    return [unif_nica_layer(n, m, k) for (n, m, k)
            in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def init_layer_params(in_dim, out_dim, key):
    W_key, b_key = jrandom.split(key, 2)
    W_init = nn.initializers.glorot_uniform(dtype=jnp.float64)
    b_init = nn.initializers.normal(dtype=jnp.float64)
    return W_init(W_key, (in_dim, out_dim)), b_init(b_key, (out_dim,))


@jit
def nica_mlp(params, s, activation='xtanh', slope=0.01):
    """Forward-pass of mixing function.
    """
    act = xtanh(slope) # add option to switch to smoothleakyrelu
    z = s

    def _fwd_pass(z, W_list):
        for i in range(len(W_list[:-1])):
            z = act(z@W_list[i])
        return z@W_list[-1]

    z = lax.cond(len(params) > 1, lambda a, B: _fwd_pass(a, B),
                 lambda a, B: a@B[0], z, params)

    #if len(params) > 1:
    #    hidden_params = params[:-1]
    #    for i in range(len(hidden_params)):
    #        z = act(z@hidden_params[i])
    #A_final = params[-1]
    #z = z@A_final
    return z


@jit
def nica_logpx(x, s, theta_x):
    theta_mix, theta_var = theta_x
    mu = nica_mlp(theta_mix, s)
    S = jnp.diag(jnp.exp(theta_var))
    return jsp.stats.multivariate_normal.logpdf(x, mu, S)




