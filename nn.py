from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jrandom
from jax import vmap


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
        A = jrandom.uniform(key, (N, M), minval=0., maxval=2.) - 1.
        A = l2normalize(A)
        _cond = jnp.linalg.cond(A)
        return A, _cond

    # generate multiple matrices
    keys = jrandom.split(key, iter_4_cond)
    A, conds = vmap(_gen_matrix, (None, None, 0))(N, M, keys)
    target_cond = jnp.percentile(conds, 25)
    target_idx = jnp.argmin(jnp.abs(conds-target_cond))
    return A[target_idx]


def init_nica_params(N, M, nonlin_layers, key, repeat_layers):
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


def nica_mlp(params, s, activation='xtanh', slope=0.1):
    """Forward-pass of mixing function.
    """
    if activation == 'xtanh':
        act = xtanh(slope)
    else:
        act = SmoothLeakyRelu(slope)
    z = s
    if len(params) > 1:
        hidden_params = params[:-1]
        for i in range(len(hidden_params)):
            z = act(z@hidden_params[i])
    A_final = params[-1]
    z = z@A_final
    return z


