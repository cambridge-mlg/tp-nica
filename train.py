from jax._src.numpy.lax_numpy import zeros_like
from jax._src.tree_util import tree_map
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
#import optax
#
import pdb
import itertools
#
from jax import vmap, jit, lax
#from jax.lax import cond
#from optax import chain, piecewise_constant_schedule, scale_by_schedule
from tprocess.kernels import (
    rdm_gamma_params,
    rdm_SE_kernel_params
)
from nn import init_nica_params
from utils import rdm_upper_cholesky_of_precision
from inference import elbo


def train(x, z, s, t, tp_mean_fn, tp_kernel_fn, params, args, est_key):
    # unpack useful args
    N = args.N
    M = args.M
    L = args.L
    T = args.T
    n_data = args.num_data
    minib_size = args.minib_size
    num_epochs = args.num_epochs

    # initialize generative model params (theta)
    key, *gamma_keys = jr.split(est_key, N+1)
    key, *k_keys = jr.split(key, N+1)
    key, Q_key = jr.split(key)
    key, mlp_key = jr.split(key)
    theta_r = vmap(lambda _: rdm_gamma_params(_, 20))(jnp.vstack(gamma_keys))
    theta_k = vmap(lambda _: rdm_SE_kernel_params(_, t))(jnp.vstack(k_keys))
    theta_Q = jnp.eye(M)*jr.uniform(Q_key, shape=(M,), minval=0.1, maxval=2.)
    theta_x = init_nica_params(N, M, L, mlp_key, repeat_layers=False)
    theta = (theta_x, theta_Q, theta_k, theta_r)

    # initialize variational parameters (phi)
    key, *v_keys = jr.split(key, T+1)
    key, *phi_r_keys = jr.split(key, n_data*N+1)
    V = vmap(lambda _: rdm_upper_cholesky_of_precision(_, N),
             out_axes=-1)(jnp.vstack(v_keys))
    phi_s = (jnp.zeros_like(x), jnp.repeat(V[None, :], n_data, 0))
    phi_r = vmap(rdm_gamma_params)(jnp.vstack(phi_r_keys))
    phi_r = tree_map(lambda _: _.reshape(n_data, N, -1), phi_r)
    phi = (phi_s, phi_r)

    # set up training
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)

    # define elbo over minibatch


    # train over minibatches
    train_data = x.copy()
    key, shuffle_key = jr.split(key)
    # train for multiple epochs
    for epoch in range(num_epochs):
        shuffle_key, shuffkey = jr.split(shuffle_key)
        shuff_data = jr.permutation(shuffkey, train_data)
        # iterate over all minibatches
        for it in range(num_minibs):
            key, it_key = jr.split(key)
            x_it = shuff_data[it*minib_size:(it+1)*minib_size]
            pdb.set_trace()


    return 0











