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
from util import rngcall
from inference import avg_neg_elbo


def train(x, z, s, t, tp_mean_fn, tp_kernel_fn, params, args, key):
    # unpack useful args
    N = args.N
    M = args.M
    L = args.L
    T = args.T
    n_data = args.num_data
    minib_size = args.minib_size
    num_epochs = args.num_epochs

    # initialize generative model params (theta)
    theta_r, key = rngcall(
        lambda _k: vmap(lambda _: jr.uniform(_, minval=2, maxval=20)
                       )(jr.split(_k, N)), key
    )
    theta_k, key = rngcall(
        lambda _k: vmap(lambda _: rdm_SE_kernel_params(_, t)
                       )(jr.split(_k, N)), key
    )
    theta_Q, key = rngcall(lambda _: jnp.eye(M)*jr.uniform(_, shape=(M,),
                minval=0.1, maxval=2.), key)
    theta_mix, key = rngcall(lambda _: init_nica_params(
        _, N, M, L, repeat_layers=False), key)
    theta_x = (theta_mix, theta_Q)
    theta = (theta_x, theta_k, theta_r)

    # initialize variational parameters (phi)
    key, *w_keys = jr.split(key, T+1)
    key, *phi_r_keys = jr.split(key, n_data*N+1)
    W = vmap(lambda _: rdm_upper_cholesky_of_precision(_, N),
             out_axes=-1)(jnp.vstack(w_keys))
    if args.diag_approx:
        W = vmap(lambda _: jnp.diag(_), in_axes=-1, out_axes=-1)(W)
    phi_s = (jnp.zeros_like(x), jnp.repeat(W[None, :], n_data, 0))
    phi_r = vmap(rdm_gamma_params)(jnp.vstack(phi_r_keys))
    phi_r = tree_map(lambda _: _.reshape(n_data, N), phi_r)
    phi = (phi_s, phi_r)

    # set up training
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)

    # define elbo over minibatch
    def avg_neg_elbo(rng, theta, phi, logpx, cov, x, t, nsamples):
        elbo = vmap(
            lambda a, b, c: elbo(a, theta, b, logpx, cov, c, t, nsamples)
        )(jr.split(rng, x.shape[0]), phi, x)
        return -elbo.mean()

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











