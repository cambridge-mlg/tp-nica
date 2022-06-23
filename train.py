from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import pdb

from jax import vmap, jit, value_and_grad
from jax.tree_util import tree_map
from functools import partial

from tprocess.kernels import rdm_SE_kernel_params, rdm_df
from nn import init_nica_params, nica_logpx
from utils import rdm_upper_cholesky_of_precision, matching_sources_corr
from utils import plot_ic
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
    nsamples = (args.num_s_samples, args.num_tau_samples)
    lr = args.learning_rate

    # initialize generative model params (theta)
    theta_tau, key = rngcall(
        lambda _k: vmap(lambda _: rdm_df(_, maxval=20))(jr.split(_k, N)), key
    )
    theta_k, key = rngcall(
        lambda _k: vmap(lambda _: rdm_SE_kernel_params(_, t)
                       )(jr.split(_k, N)), key
    )
    theta_var, key = rngcall(lambda _: jr.uniform(_, shape=(M,),
                minval=-1, maxval=1), key)
    theta_mix, key = rngcall(lambda _: init_nica_params(
        _, N, M, L, repeat_layers=False), key)
    theta_x = (theta_mix, theta_var)
    theta = (theta_x, theta_k, theta_tau)

    # initialize variational parameters (phi) with pseudo-points (tu)
    tu, key = rngcall(lambda k: jr.uniform(k, shape=(n_data, n_pseudo, 1),
                                           minval=jnp.min(t),
                                           maxval=jnp.max(t)), key)
    W, key = rngcall(lambda _k: vmap(
        lambda _: rdm_upper_cholesky_of_precision(_, N), out_axes=-1)(
            jr.split(_k, n_pseudo)), key
    )
    if args.diag_approx:
        W = vmap(lambda _: jnp.diag(_), in_axes=-1, out_axes=-1)(W)
    phi_s = (jnp.repeat(W[None, :], n_data, 0),
             jnp.ones(shape=(n_data, N, n_pseudo)), tu)
    phi_df, key = rngcall(lambda _: vmap(rdm_df)(jr.split(_, n_data*N)), key)
    phi_df = phi_df.reshape(n_data, N)
    phi_tau = (phi_df, phi_df)
    phi = (phi_s, phi_tau)

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
            (nvlb, s), g = value_and_grad(avg_neg_elbo, argnums=(1, 2),
                                     has_aux=True)(key, theta, phi_n, logpx,
                                                   kernel_fn, x, t, nsamples)
            s = s.mean(axis=(1,2)).swapaxes(-1, -2)
            theta_g, phi_n_g = g

            # perform gradient updates
            theta_updates, theta_opt_state = optimizer.update(
                theta_g, theta_opt_state, theta)
            theta = optax.apply_updates(theta, theta_updates)
            phi_n_updates, phi_n_opt_states = vmap(optimizer.update)(
                phi_n_g, phi_n_opt_states, phi_n)
            phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
            return nvlb, s, theta, phi_n, theta_opt_state, phi_n_opt_states
        return training_step


    training_step = make_training_step(nica_logpx, tp_kernel_fn, t,
                                       nsamples)

    # train over minibatches
    train_data = x.copy()
    s_data = s.copy()
    # train for multiple epochs
    for epoch in range(num_epochs):
        shuffle_idx, key = rngcall(jr.permutation, key, n_data)
        shuff_data = train_data[shuffle_idx]
        shuff_s = s_data[shuffle_idx]
        # iterate over all minibatches
        for it in range(num_minibs):
            x_it = shuff_data[it*minib_size:(it+1)*minib_size]
            s_it = shuff_s[it*minib_size:(it+1)*minib_size]
            # select variational parameters of the observations in minibatch
            idx_set_it = shuffle_idx[it*minib_size:(it+1)*minib_size]
            phi_it = tree_get_idx(phi, idx_set_it)
            phi_opt_states_it = tree_get_idx(phi_opt_states, idx_set_it)

            # training step
            (nvlb, s_sample, theta, phi_it, theta_opt_state, phi_opt_states_it), key = rngcall(
                training_step, key, theta, phi_it, theta_opt_state,
                phi_opt_states_it, x_it)

            # update the full variational parameter pytree at right indices
            phi = tree_map(lambda a, b: a.at[idx_set_it].set(b), phi, phi_it)
            phi_opt_states = tree_map(lambda a, b: a.at[idx_set_it].set(b),
                                      phi_opt_states, phi_opt_states_it)

            # evaluate
            minib_mccs = []
            for j in range(minib_size):
                mcc, _, sort_idx = matching_sources_corr(s_sample[j], s_it[j])
                minib_mccs.append(mcc)
            minib_avg_mcc = jnp.mean(jnp.array(minib_mccs))

            print("*Epoch: [{0}/{1}]\t"
                  "Minibatch: [{2}/{3}]\t"
                  "ELBO: {4}\t"
                  "MCC: {5}".format(epoch, num_epochs-1, it,
                                    num_minibs-1, -nvlb, minib_avg_mcc))

            # plot regularly
            if epoch % args.plot_freq == 0:
                plot_idx = 0 # which data sample to plot in each minibatch
                plot_start = 0
                plot_len = 200
                plot_end = plot_start+plot_len

                # set plot
                fig, ax = plt.subplots(1, N, figsize=(10 * N, 6), sharex=True)

                # create separate plot for each IC
                for n in range(N):
                    s_sample_n = s_sample[plot_idx][sort_idx][n, plot_start:
                                                              plot_end]
                    s_it_n = s_it[plot_idx][sort_idx][n, plot_start:
                                                      plot_end]
                    ax[n].clear()
                    ax2_n = ax[n].twinx()
                    ax[n].plot(s_it_n, color='blue')
                    ax[n].set_xlim([0, T])
                    ax2_n.plot(s_sample_n, color='red')
                plt.show(block=False)
                plt.pause(10.)
                plt.close()
    return 0





