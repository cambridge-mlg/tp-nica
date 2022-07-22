import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import pdb
import time

from jax import vmap, jit, value_and_grad, lax
from jax.tree_util import tree_map
from functools import partial

from kernels import rdm_SE_kernel_params, rdm_df
from nn import init_nica_params, nica_logpx
from utils import rdm_upper_cholesky, matching_sources_corr
from utils import plot_ic, jax_print
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
    theta_lr = args.theta_learning_rate
    phi_lr = args.phi_learning_rate
    gt_Q, gt_mixer_params, gt_kernel_params, gt_tau = params

    # initialize generative model params (theta)
    theta_tau, key = rngcall(
        lambda _k: vmap(lambda _: rdm_df(_, maxval=4))(jr.split(_k, N)), key
    )
    theta_k, key = rngcall(
        lambda _k: vmap(lambda _: rdm_SE_kernel_params(_)
                       )(jr.split(_k, N)), key
    )
    theta_var, key = rngcall(lambda _: jr.uniform(_, shape=(M,),
                minval=-1, maxval=1), key)
    theta_mix, key = rngcall(lambda _: init_nica_params(
        _, N, M, L, repeat_layers=False), key)
    theta_x = (theta_mix, theta_var)

    # for debugging: set some params to ground-truths
    if args.use_gt_nica:
        theta_x = (gt_mixer_params, jnp.log(jnp.diag(gt_Q)))
    if args.use_gt_kernel:
        theta_k = gt_kernel_params
    if args.use_gt_tau:
        theta_tau = gt_tau
    use_gt_settings = (args.use_gt_nica, args.use_gt_kernel, args.use_gt_tau)

    theta = (theta_x, theta_k, theta_tau)

    # initialize variational parameters (phi) with pseudo-points (tu)
    tu, key = rngcall(lambda _: vmap(lambda k: jr.choice(k, t,
            shape=(n_pseudo,), replace=False))(jr.split(_, n_data)), key)
    W, key = rngcall(lambda _k: vmap(
        lambda _: rdm_upper_cholesky(_, N)[jnp.triu_indices(N)]*10,
        out_axes=-1)(jr.split(_k, n_pseudo)), key
    )
    if args.diag_approx:
        W = vmap(lambda _: jnp.diag(_), in_axes=-1, out_axes=-1)(W)
    phi_s = (jnp.repeat(W[None, :], n_data, 0),
             jnp.ones(shape=(n_data, N, n_pseudo)), tu)
    phi_df, key = rngcall(lambda _: vmap(rdm_df)(jr.split(_, n_data*N)), key)
    phi_df = phi_df.reshape(n_data, N)
    phi_tau = (phi_df, phi_df*10)
    phi = (phi_s, phi_tau)

    # set up training details
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    theta_optimizer = optax.adam(theta_lr)
    phi_optimizer = optax.adam(phi_lr)
    theta_opt_state = theta_optimizer.init(theta)
    phi_opt_states = vmap(phi_optimizer.init)(phi)


    def make_training_step(logpx, kernel_fn, t, nsamples, use_gt_settings,
                           optimizers):
        @jit
        def training_step(key, theta, phi_n, theta_opt_state,
                          phi_n_opt_states, x):
            (nvlb, s), g = value_and_grad(avg_neg_elbo, argnums=(1, 2),
                                     has_aux=True)(key, theta, phi_n, logpx,
                                                   kernel_fn, x, t, nsamples)
            s = s.mean(axis=(1, 2)).swapaxes(-1, -2)
            theta_g, phi_n_g = g

            # perform gradient updates
            (theta_opt, phi_opt) = optimizers
            theta_updates, theta_opt_state = theta_opt.update(
                theta_g, theta_opt_state, theta)

            # override updates in debug mode
            use_gt_nica, use_gt_kernel, use_gt_tau = use_gt_settings
            gt_override_fun = lambda x: tree_map(lambda _:
                            jnp.zeros(shape=_.shape), x)
            nica_updates = lax.cond(use_gt_nica, gt_override_fun,
                        lambda x: x, theta_updates[0])
            kernel_updates = lax.cond(use_gt_kernel, gt_override_fun,
                        lambda x: x, theta_updates[1])
            tau_updates = lax.cond(use_gt_tau, gt_override_fun,
                        lambda x: x, theta_updates[2])
            theta_updates = (nica_updates, kernel_updates, tau_updates)

            # perform gradient updates
            theta = optax.apply_updates(theta, theta_updates)
            phi_n_updates, phi_n_opt_states = vmap(phi_opt.update)(
                phi_n_g, phi_n_opt_states, phi_n)
            phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
            return nvlb, s, theta, phi_n, theta_opt_state, phi_n_opt_states
        return training_step


    training_step = make_training_step(nica_logpx, tp_kernel_fn, t,
                                       nsamples, use_gt_settings,
                                       (theta_optimizer, phi_optimizer))

    # train over minibatches
    train_data = x.copy()
    s_data = s.copy()
    mcc_hist = []
    elbo_hist = []
    # train for multiple epochs
    for epoch in range(num_epochs):
        tic = time.perf_counter()
        shuffle_idx, key = rngcall(jr.permutation, key, n_data)
        shuff_data = train_data[shuffle_idx]
        shuff_s = s_data[shuffle_idx]
        mcc_epoch_hist = []
        elbo_epoch_hist = []
        # iterate over all minibatches
        for it in range(num_minibs):
            x_it = shuff_data[it*minib_size:(it+1)*minib_size]
            s_it = shuff_s[it*minib_size:(it+1)*minib_size]
            # select variational parameters of the observations in minibatch
            idx_set_it = shuffle_idx[it*minib_size:(it+1)*minib_size]
            phi_it = tree_get_idx(phi, idx_set_it)
            phi_opt_states_it = tree_get_idx(phi_opt_states, idx_set_it)

            # training step
            (nvlb, s_sample, theta, phi_it, theta_opt_state,
             phi_opt_states_it), key = rngcall(
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
            mcc_epoch_hist.append(minib_avg_mcc.item())
            elbo_epoch_hist.append(-nvlb.item())

            print("*Epoch: [{0}/{1}]\t"
                  "Minibatch: [{2}/{3}]\t"
                  "ELBO: {4}\t"
                  "MCC: {5}".format(epoch, num_epochs-1, it,
                                    num_minibs-1, -nvlb, minib_avg_mcc))

            ## plot regularly
            if epoch % args.plot_freq == 0 and it == 0:
                plot_idx = 0 # which data sample to plot in each minibatch
                plot_start = 0
                plot_len = min(1000, T)
                plot_end = plot_start+plot_len

                # set plot
                fig, ax = plt.subplots(1, N, figsize=(10 * N, 6), sharex=True)

                # create separate plot for each IC
                for n in range(N):
                    s_sample_n = s_sample[plot_idx][sort_idx][n, plot_start:
                                                              plot_end]
                    s_it_n = s_it[plot_idx][n, plot_start:plot_end]
                    ax[n].clear()
                    ax2_n = ax[n].twinx()
                    ax[n].plot(s_it_n, color='blue')
                    ax[n].set_xlim([plot_start, plot_end])
                    ax2_n.plot(s_sample_n, color='red')
                if args.headless:
                    plt.savefig("s_vs_sest.png")
                else:
                    plt.show(block=False)
                    plt.pause(10.)
                plt.close()
        toc = time.perf_counter()
        epoch_avg_mcc = jnp.mean(jnp.array(mcc_epoch_hist))
        epoch_avg_elbo = jnp.mean(jnp.array(elbo_epoch_hist))
        mcc_hist.append(epoch_avg_mcc.item())
        elbo_hist.append(epoch_avg_elbo.item())

        print("Epoch took: {0}\t"
              "AVG. ELBO: {1} \t"
              "AVG. MCC: {2}".format(toc-tic, epoch_avg_elbo, epoch_avg_mcc))
    return mcc_hist, elbo_hist
