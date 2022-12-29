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


def unif_nica_layer(N, M, key, iter_4_cond=10000):
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
    # NOTE: bias has been fixed to zero for now:
    b = jrandom.uniform(jrandom.split(keys[-1])[0], (M,), minval=0., maxval=0.)
    return A[target_idx], b


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


@jit
def nica_mlp(params, s, activation='xtanh', slope=0.01):
    """Forward-pass of mixing function.
    """
    def _fwd_pass(z, params_list):
        for A, b in params_list[:-1]:
            z = act(z@A)+b
        return z@params_list[-1][0]+params_list[-1][1]


    act = xtanh(slope) # add option to switch to smoothleakyrelu
    z = s
    z = lax.cond(len(params) > 1, lambda a, B: _fwd_pass(a, B),
                 lambda a, B: a@B[0][0]+B[0][1], z, params)
    return z


@jit
def nica_logpx(x, s, theta_x):
    theta_mix, theta_var = theta_x
    mu = nica_mlp(theta_mix, s)
    S = jnp.diag(jnp.exp(theta_var))
    return jsp.stats.multivariate_normal.logpdf(x, mu, S)




