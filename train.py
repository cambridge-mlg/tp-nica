from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax

import pdb

from jax import vmap, jit, value_and_grad
from jax.tree_util import tree_map
from functools import partial

from tprocess.kernels import rdm_SE_kernel_params, rdm_df
from nn import init_nica_params, nica_logpx
from utils import rdm_upper_cholesky_of_precision
from util import rngcall, tree_get_idx, tree_get_range
from inference import avg_neg_elbo


def train(x, z, s, t, tp_mean_fn, tp_kernel_fn, params, args, key):
    # unpack useful args
    N = args.N
    M = args.M
    L = args.L
    T = args.T
    n_data = args.num_data
    n_pseudo = args.num_pseudo
    minib_size = args.minib_size
    num_epochs = args.num_epochs
    nsamples = (args.num_r_samples, args.num_s_samples)
    lr = args.learning_rate

    # initialize generative model params (theta)
    theta_tau, key = rngcall(
        lambda _k: vmap(lambda _: rdm_df(_, maxval=20))(jr.split(_k, N)), key
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
    theta = (theta_x, theta_k, theta_tau)

    # initialize variational parameters (phi) with pseudo-points (tu)
    tu, key = rngcall(lambda k: jr.uniform(k, shape=(n_data, n_pseudo, 1),
                                           minval=jnp.min(t),
                                           maxval=jnp.max(t)), key)
    W, key = rngcall(lambda _k: vmap(
        lambda _: rdm_upper_cholesky_of_precision(_, N), out_axes=-1)(
            jr.split(_k, len(tu))), key
    )
    if args.diag_approx:
        W = vmap(lambda _: jnp.diag(_), in_axes=-1, out_axes=-1)(W)
    phi_s = (jnp.zeros(shape=(n_data, N, len(tu))),
             jnp.repeat(W[None, :], n_data, 0), tu)
    phi_df, key = rngcall(lambda _: vmap(rdm_df)(jr.split(_, n_data*N)), key)
    phi_df = phi_df.reshape(n_data, N)
    phi_tau = (phi_df, phi_df)
    phi = (phi_s, phi_tau)
    pdb.set_trace()

    # set up training params
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    optimizer = optax.adam(lr)
    phi_opt_states = vmap(optimizer.init)(phi)
    theta_opt_state = optimizer.init(theta)


    def make_training_step(logpx, kernel_fn, t, nsamples):
        #@jit
        def training_step(key, theta, phi_n, theta_opt_state,
                          phi_n_opt_states, x):
            nvlb, g = value_and_grad(avg_neg_elbo, argnums=(1, 2))(
                              key, theta, phi_n, logpx,
                              kernel_fn, x, t, nsamples)
            theta_g, phi_n_g = g

            # perform gradient updates
            theta_updates, theta_opt_state = optimizer.update(
                theta_g, theta_opt_state, theta)
            theta = optax.apply_updates(theta, theta_updates)
            phi_n_updates, phi_n_opt_states = vmap(optimizer.update)(
                phi_n_g, phi_n_opt_states, phi_n)
            phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
            return nvlb, theta, phi_n, theta_opt_state, phi_n_opt_states
        return training_step


    training_step = make_training_step(nica_logpx, tp_kernel_fn, t,
                                       nsamples)

    # train over minibatches
    train_data = x.copy()
    # train for multiple epochs
    for epoch in range(num_epochs):
        shuffle_idx, key = rngcall(jr.permutation, key, n_data)
        shuff_data = train_data[shuffle_idx]
        # iterate over all minibatches
        for it in range(num_minibs):
            x_it = shuff_data[it*minib_size:(it+1)*minib_size]
            # select variational parameters of the observations in minibatch
            idx_set_it = shuffle_idx[it*minib_size:(it+1)*minib_size]
            phi_it = tree_get_idx(phi, idx_set_it)
            phi_opt_states_it = tree_get_idx(phi_opt_states, idx_set_it)

            # training step
            (nvlb, theta, phi_it, theta_opt_state, phi_opt_states_it), key = rngcall(
                training_step, key, theta, phi_it, theta_opt_state,
                phi_opt_states_it, x_it)

            # update the full variational parameter pytree at right indices
            phi = tree_map(lambda a, b: a.at[idx_set_it].set(b), phi, phi_it)
            phi_opt_states = tree_map(lambda a, b: a.at[idx_set_it].set(b),
                                      phi_opt_states, phi_opt_states_it)


            print("*Epoch: [{0}/{1}]\t"
                  "Minibatch: [{2}/{3}]\t"
                  "ELBO: {4}".format(epoch, num_epochs, it, num_minibs, -nvlb))
    return 0





