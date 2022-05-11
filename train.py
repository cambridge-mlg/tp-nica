from jax.tree_util import tree_map
from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
#import optax
#
import pdb
#
from jax import vmap, jit, lax
#from jax.lax import cond
#from optax import chain, piecewise_constant_schedule, scale_by_schedule
from tprocess.kernels import rdm_SE_kernel_params, rdm_df, se_kernel_fn
from nn import init_nica_params
from utils import rdm_upper_cholesky_of_precision
from util import rngcall
from inference import avg_neg_elbo
from nn import nica_mlp, nica_logpx


def train(x, z, s, t, tp_mean_fn, tp_kernel_fn, params, args, key):
    # unpack useful args
    N = args.N
    M = args.M
    L = args.L
    T = args.T
    n_data = args.num_data
    minib_size = args.minib_size
    num_epochs = args.num_epochs
    nsamples = (args.num_r_samples, args.num_s_samples)

    # initialize generative model params (theta)
    theta_r, key = rngcall(
        lambda _k: vmap(lambda _: rdm_df(_, maxval=20)
                       )(jr.split(_k, N)), key
    )
    theta_k, key = rngcall(
        lambda _k: vmap(lambda _: rdm_SE_kernel_params(_, t)
                       )(jr.split(_k, N)), key
    )
    theta_Q, key = rngcall(lambda _: jnp.eye(M)*jr.uniform(_, shape=(M,),
                minval=0.01, maxval=0.2), key)
    theta_mix, key = rngcall(lambda _: init_nica_params(
        _, N, M, L, repeat_layers=False), key)
    theta_x = (theta_mix, theta_Q)
    theta = (theta_x, theta_k, theta_r)

    # initialize variational parameters (phi)
    W, key = rngcall(lambda _k: vmap(
        lambda _: rdm_upper_cholesky_of_precision(_, N), out_axes=-1)(
            jr.split(_k, T)), key
    )
    if args.diag_approx:
        W = vmap(lambda _: jnp.diag(_), in_axes=-1, out_axes=-1)(W)
    phi_s = (jnp.zeros_like(x), jnp.repeat(W[None, :], n_data, 0))
    phi_nu, key = rngcall(lambda _: vmap(rdm_df)(jr.split(_, n_data*N)), key)
    phi_r = (phi_nu, phi_nu)
    phi_r = tree_map(lambda _: _.reshape(n_data, N), phi_r)
    phi = (phi_s, phi_r)

    # initialize likelihood function
    logpx = lambda _: nica_logpx(x, s, _)


    # set up training
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)

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





