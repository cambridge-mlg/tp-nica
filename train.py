import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import pdb
import time
import os
import cloudpickle

from jax import vmap, jit, value_and_grad, lax
from jax.tree_util import tree_map

from kernels import rdm_SE_kernel_params, rdm_df
from nn import init_nica_params, nica_logpx
from utils import (
    sample_wishart,
    matching_sources_corr,
    save_checkpoint,
    load_checkpoint
)

from util import rngcall, tree_get_idx
from inference import avg_neg_tp_elbo, avg_neg_gp_elbo


def train(x, z, s, t, tp_mean_fn, tp_kernel_fn, params, args, key):
    # unpack useful args
    N = args.N
    M = args.M
    L = args.L_est
    T = args.T
    n_data = args.num_data
    minib_size = args.minib_size
    num_epochs = args.num_epochs
    theta_lr = args.theta_learning_rate
    phi_lr = args.phi_learning_rate
    if args.GP:
        nsamples = args.num_s_samples
        gt_Q, gt_mixer_params, gt_kernel_params = params
    else:
        nsamples = (args.num_s_samples, args.num_tau_samples)
        gt_Q, gt_mixer_params, gt_kernel_params, gt_tau = params

    # initialize generative model params (theta)
    if not args.GP:
        theta_tau, key = rngcall(
            lambda _k: vmap(lambda _: rdm_df(_, maxval=4))(jr.split(_k, N)), key
        )
        theta_tau = jnp.log(theta_tau)
        if args.repeat_dfs:
            theta_tau = theta_tau[:1]

    if args.D == 1:
        theta_k, key = rngcall(
            lambda _k: vmap(lambda _: rdm_SE_kernel_params(_))(jr.split(_k, N)),
            key
        )
    elif args.D == 2:
        theta_k, key = rngcall(
            lambda _k: vmap(lambda _: rdm_SE_kernel_params(
                _, min_lscale=4., max_lscale=10.))(jr.split(_k, N)),
            key
        )
    theta_k = tree_map(lambda _: jnp.log(_), theta_k)
    if args.repeat_kernels:
        theta_k = tree_map(lambda _: _[:1], theta_k)

    theta_var, key = rngcall(lambda _: jr.uniform(_, shape=(M,),
                minval=-1, maxval=1), key)
    theta_mix, key = rngcall(lambda _: init_nica_params(
        _, N, M, L, repeat_layers=False), key)
    theta_x = (theta_mix, theta_var)

    # for debugging: set some params to ground-truths
    if args.use_gt_nica:
        theta_x = (gt_mixer_params, jnp.log(jnp.diag(gt_Q)))
    if args.use_gt_kernel:
        theta_k = tree_map(lambda _:  jnp.log(_), gt_kernel_params)
    if args.use_gt_tau and not args.GP:
        theta_tau = jnp.log(gt_tau)

    if args.GP:
        use_gt_settings = (args.use_gt_nica, args.use_gt_kernel)
        theta = (theta_x, theta_k)
    else:
        use_gt_settings = (args.use_gt_nica, args.use_gt_kernel, args.use_gt_tau)
        theta = (theta_x, theta_k, theta_tau)

    # initialize variational parameters (phi)
    L, key = rngcall(lambda _k: vmap(
        lambda _: jnp.linalg.cholesky(
            sample_wishart(_, jnp.array(N+1.), 10*jnp.eye(N))
        )[jnp.tril_indices(N)],
        out_axes=-1)(jr.split(_k, T)), key
    )
    phi_s = (jnp.repeat(L[None, :], n_data, 0), jnp.ones((n_data, N, T)))
    if args.GP:
        phi = phi_s
    else:
        phi_df, key = rngcall(lambda _:vmap(rdm_df)(jr.split(_, n_data*N)), key)
        phi_df = jnp.log(phi_df.reshape(n_data, N))
        phi_tau = (phi_df, phi_df)
        phi = (phi_s, phi_tau)

    # set up training details
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    theta_optimizer = optax.adam(theta_lr)
    phi_optimizer = optax.adam(phi_lr)
    theta_opt_state = theta_optimizer.init(theta)
    phi_opt_states = vmap(phi_optimizer.init)(phi)

    # optionally load from checkpoint
    if args.resume_ckpt:
        ckpt, hist = load_checkpoint(args)
        ckpt_epoch, key, theta, phi, theta_opt_state, phi_opt_states = ckpt
        elbo_hist, mcc_hist = hist


    # define training step
    def make_training_step(logpx, kernel_fn, t, nsamples, use_gt_settings,
                           optimizers, is_gp):
        if is_gp:
            @jit
            def gp_training_step(key, theta, phi_n, theta_opt_state,
                                 phi_n_opt_states, x, burn_in):
                (nvlb, s), g = value_and_grad(avg_neg_gp_elbo, argnums=(1, 2),
                                         has_aux=True)(key, theta, phi_n, logpx,
                                                       kernel_fn, x, t, nsamples)
                theta_g, phi_n_g = g

                # set up gradient updates for theta
                (theta_opt, phi_opt) = optimizers
                theta_updates, theta_opt_state = theta_opt.update(
                    theta_g, theta_opt_state, theta)

                # override theta updates in debug mode with gt params
                use_gt_nica, use_gt_kernel = use_gt_settings
                grad_override_fun = lambda x: tree_map(
                    lambda _: jnp.zeros(shape=_.shape), x)
                nica_updates = lax.cond(use_gt_nica, grad_override_fun,
                            lambda x: x, theta_updates[0])
                kernel_updates = lax.cond(use_gt_kernel, grad_override_fun,
                            lambda x: x, theta_updates[1])
                theta_updates = (nica_updates, kernel_updates)

                # also stop updates during burn-in
                theta_updates = lax.cond(burn_in, grad_override_fun,
                                         lambda x: x, theta_updates)

                # perform gradient updates
                theta = optax.apply_updates(theta, theta_updates)
                phi_n_updates, phi_n_opt_states = vmap(phi_opt.update)(
                    phi_n_g, phi_n_opt_states, phi_n)
                phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
                return nvlb, s, theta, phi_n, theta_opt_state, phi_n_opt_states
            return gp_training_step
        else:
            #@jit
            def tp_training_step(key, theta, phi_n, theta_opt_state,
                              phi_n_opt_states, x, burn_in):
                (nvlb, s), g = value_and_grad(avg_neg_tp_elbo, argnums=(1, 2),
                                         has_aux=True)(key, theta, phi_n, logpx,
                                                       kernel_fn, x, t, nsamples)
                theta_g, phi_n_g = g

                # set up gradient updates for theta 
                (theta_opt, phi_opt) = optimizers
                theta_updates, theta_opt_state = theta_opt.update(
                    theta_g, theta_opt_state, theta)

                # override updates in debug mode
                use_gt_nica, use_gt_kernel, use_gt_tau = use_gt_settings
                grad_override_fun = lambda x: tree_map(
                    lambda _: jnp.zeros(shape=_.shape), x)
                nica_updates = lax.cond(use_gt_nica, grad_override_fun,
                            lambda x: x, theta_updates[0])
                kernel_updates = lax.cond(use_gt_kernel, grad_override_fun,
                            lambda x: x, theta_updates[1])
                tau_updates = lax.cond(use_gt_tau, grad_override_fun,
                            lambda x: x, theta_updates[2])
                theta_updates = (nica_updates, kernel_updates, tau_updates)

                # stop updates during burn-in 
                theta_updates = lax.cond(burn_in, grad_override_fun,
                                         lambda x: x, theta_updates)

                # perform gradient updates
                theta = optax.apply_updates(theta, theta_updates)
                phi_n_updates, phi_n_opt_states = vmap(phi_opt.update)(
                    phi_n_g, phi_n_opt_states, phi_n)
                phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
                return nvlb, s, theta, phi_n, theta_opt_state, phi_n_opt_states
            return tp_training_step


    # define evaluation step
    def make_eval_step(logpx, kernel_fn, t, nsamples, elbo_fn):
        def eval_step(key, theta, phi_n, x):
            (nvlb, s)  = elbo_fn(key, theta, phi_n, logpx,
                                      kernel_fn, x, t, nsamples)
            return nvlb, s
        return eval_step


    # initialize eval/training step
    if args.eval_only:
        if args.GP:
            elbo_fn = avg_neg_gp_elbo
        else:
            elbo_fn = avg_neg_tp_elbo
        eval_step = make_eval_step(nica_logpx, tp_kernel_fn, t,
                                   nsamples, elbo_fn)
    else:
        training_step = make_training_step(
            nica_logpx, tp_kernel_fn, t ,nsamples, use_gt_settings,
            (theta_optimizer, phi_optimizer), args.GP
        )

    # set up training
    train_data = x.copy()
    s_data = s.copy()
    if args.resume_ckpt:
        start_epoch = ckpt_epoch+1
        if args.eval_only:
            num_epochs = 1
    else:
        start_epoch = 0
        mcc_hist = []
        elbo_hist = []
    best_elbo = -jnp.inf
    # train for multiple epochs
    for epoch in range(start_epoch, num_epochs):
        tic = time.perf_counter()
        shuffle_idx, key = rngcall(jr.permutation, key, n_data)
        shuff_data = train_data[shuffle_idx]
        shuff_s = s_data[shuffle_idx]
        mcc_epoch_hist = []
        elbo_epoch_hist = []
        burn_in = epoch < args.burn_in_len
        # iterate over all minibatches
        for it in range(num_minibs):
            x_it = shuff_data[it*minib_size:(it+1)*minib_size]
            s_it = shuff_s[it*minib_size:(it+1)*minib_size]
            # select variational parameters of the observations in minibatch
            idx_set_it = shuffle_idx[it*minib_size:(it+1)*minib_size]
            phi_it = tree_get_idx(phi, idx_set_it)
            phi_opt_states_it = tree_get_idx(phi_opt_states, idx_set_it)

            # training step (or evaluation)
            if args.eval_only:
                nvlb, s_sample = eval_step(key, theta, phi_it, x_it)
            else:
                (nvlb, s_sample, theta, phi_it, theta_opt_state,
                 phi_opt_states_it), key = rngcall(
                    training_step, key, theta, phi_it, theta_opt_state,
                    phi_opt_states_it, x_it, burn_in)
 
                # update the full variational parameter pytree at right indices
                phi = tree_map(lambda a, b: a.at[idx_set_it].set(b), phi, phi_it)
                phi_opt_states = tree_map(lambda a, b: a.at[idx_set_it].set(b),
                                          phi_opt_states, phi_opt_states_it)

            # evaluate
            s_sample = s_sample.swapaxes(-1, -2)
            if args.GP:
                s_sample = s_sample.mean(axis=(1,))
            else:
                s_sample = s_sample.mean(axis=(1, 2))
            minib_mccs = []
            for j in range(minib_size):
                mcc, _, sort_idx = matching_sources_corr(s_sample[j], s_it[j])
                minib_mccs.append(mcc)
            minib_avg_mcc = jnp.mean(jnp.array(minib_mccs))
            mcc_epoch_hist.append(minib_avg_mcc.item())
            elbo_epoch_hist.append(-nvlb.item())

            print("*Epoch: [{0}/{1}]\t"
                  "Minibatch: [{2}/{3}]\t"
                  "ELBO: {4:.2f}\t"
                  "MCC: {5:.3f}".format(epoch, num_epochs-1, it,
                                        num_minibs-1, -nvlb, minib_avg_mcc))

            ## plot regularly
            if (epoch % args.plot_freq == 0 or args.eval_only) and it == 0:
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

        print("Epoch [{0}/{1}] took: {2:.2f}\t"
              "AVG. ELBO: {3:.2f}\t"
              "AVG. MCC: {4:.3f}\t"
              "data seed: {5}\t"
              "est. seed: {6}\t"
              "init. theta lr: {7}\t"
              "init. phi lr: {8}".format(epoch, num_epochs-1,
                                         toc-tic, epoch_avg_elbo,
                                         epoch_avg_mcc, args.data_seed,
                                         args.est_seed, theta_lr, phi_lr))

        # save checkpoints
        if not args.eval_only:
            mcc_hist.append(epoch_avg_mcc.item())
            elbo_hist.append(epoch_avg_elbo.item())
            if epoch_avg_elbo > best_elbo:
                print("**Saving checkpoint (best elbo thus far)**")
                best_elbo = epoch_avg_elbo
                save_checkpoint((epoch, key, theta, phi, theta_opt_state,
                                 phi_opt_states), (elbo_hist, mcc_hist), args)

        # plot training histories
        if epoch % args.plot_freq == 0 or args.eval_only:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(elbo_hist)
            ax2.plot(mcc_hist)
            plt.tight_layout()
            if args.headless:
                plt.savefig("elbo_mcc_hist.png")
            else:
                plt.show(block=False)
                plt.pause(5.)
            plt.close()
    return elbo_hist, mcc_hist
