from jax.tree_util import tree_map
from jax.config import config
import optax

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
#import optax
#
import pdb
#
from jax import vmap, jit, lax, value_and_grad
from optax import chain, piecewise_constant_schedule, scale_by_schedule

from tprocess.kernels import rdm_SE_kernel_params, rdm_df
from nn import init_nica_params, nica_logpx
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
    nsamples = (args.num_r_samples, args.num_s_samples)
    lr = args.learning_rate

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
    params = (theta, phi)

    # initialize likelihood function
    logpx = lambda _: nica_logpx(x, s, _)

    # set up training params
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)


    def training_step(key, opt_state, params, logpx, kernel_fn, x, t, nsamples):
        theta, phi = params
        (nvlb, s), g = value_and_grad(
            avg_neg_elbo, argnums=(1, 2), has_aux=True)(
                key, theta, phi, logpx, kernel_fn, x, t, nsamples
            )

        # perform gradient updates
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        return nvlb, s, params, opt_state


    # train over minibatches
    train_data = x.copy()
    key, shuffle_key = jr.split(key)
    # train for multiple epochs
    for epoch in range(num_epochs):
        shuffle_key, shuffkey = jr.split(shuffle_key)
        shuff_data = jr.permutation(shuffkey, train_data)
        # iterate over all minibatches
        for it in range(num_minibs):
            key, tr_key = jr.split(key)
            x_it = shuff_data[it*minib_size:(it+1)*minib_size]
            nvlb, s, params, opt_state = training_step(
                tr_key, opt_state, params, logpx, tp_kernel_fn, x, t,
                nsamples)

    return 0





