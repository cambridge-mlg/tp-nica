import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import matplotlib.pyplot as plt
import pdb
import time
import os
import cloudpickle
import jax.debug as jdb

from jax import vmap, jit, value_and_grad, lax
from jax.tree_util import tree_map
from scipy.cluster.vq import kmeans2

from kernels import rdm_SE_kernel_params, rdm_df
from nn import init_nica_params, nica_logpx
from utils import (
    sample_wishart,
    tree_zeros_like,
    _identity,
    matching_sources_corr,
    save_checkpoint,
    load_checkpoint
)

from util import rngcall, tree_get_idx
from inference import avg_neg_tp_elbo, avg_neg_gp_elbo


def train(x, t, mean_fn, kernel_fn, args, key):
    # unpack useful args
    N = args.N
    M = x.shape[1]
    L = args.L_est
    T = t.shape[0]
    n_data = x.shape[0]
    n_pseudo = args.num_pseudo
    minib_size = args.minib_size
    num_epochs = args.num_epochs
    theta_lr = args.theta_learning_rate
    phi_lr = args.phi_learning_rate
    if args.GP:
        nsamples = args.num_s_samples
        fix_df = None
    else:
        nsamples = (args.num_s_samples, args.num_tau_samples)
        fix_df = args.fix_df

    # initialize generative model params (theta)
    if not args.GP:
        theta_tau, key = rngcall(
            lambda _k: vmap(lambda _: rdm_df(_, min_val=args.tp_df,
                                             max_val=args.tp_df+2))(
                                                 jr.split(_k, N)
                                             ), key
        )
        theta_tau = jnp.log(theta_tau-2)

    theta_k, key = rngcall(
            lambda _k: vmap(rdm_SE_kernel_params,
                            in_axes=(0, None, None, None, None))(
                jr.split(_k, N), 1., 1., 1., 1.), key
        )
    theta_k = tree_map(jnp.log, theta_k)
    theta_var, key = rngcall(lambda _: jr.uniform(_, shape=(M,),
                minval=-1, maxval=1), key)

    theta_mix, key = rngcall(lambda _: init_nica_params(
        _, N, M, L, repeat_layers=False), key)
    theta_x = (theta_mix, theta_var)

    if args.GP:
        theta = (theta_x, theta_k)
    else:
        theta = (theta_x, theta_k, theta_tau)

    # initialize variational parameters (phi) with pseudo-points (tu)
    tu = kmeans2(t, k=n_pseudo, minit='points')[0]
    tu = jnp.tile(tu, (n_data, 1, 1))

    W, key = rngcall(
        lambda _k: vmap(lambda _: jnp.linalg.cholesky(
            sample_wishart(_, jnp.array(N+1.), jnp.eye(N))
        )[jnp.tril_indices(N)], out_axes=-1)(jr.split(_k, n_pseudo)), key
    )

    phi_s = (jnp.repeat(W[None, :], n_data, 0),
             jnp.ones(shape=(n_data, N, n_pseudo)), tu)
    if args.GP:
        phi = phi_s
    else:
        phi_df, key = rngcall(lambda _: vmap(lambda _k: rdm_df(
            _k, min_val=args.tp_df, max_val=args.tp_df))(jr.split(_, n_data*N)), key)
        phi_df = jnp.log(phi_df.reshape(n_data, N))
        phi_tau = (phi_df, phi_df)
        phi = (phi_s, phi_tau)

    ## set up training details
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
        elbo_hist = hist


    # define training step
    def make_training_step(logpx, kernel_fn, t, nsamples, optimizers, is_gp,
                           fix_df):
        if is_gp:
            @jit
            def gp_training_step(key, theta, phi_n, theta_opt_state,
                                 phi_n_opt_states, x, burn_in):
                (nvlb, s), g = value_and_grad(avg_neg_gp_elbo, argnums=(1, 2),
                                         has_aux=True)(key, theta, phi_n, logpx,
                                                       kernel_fn, x, t, nsamples)
                theta_g, phi_n_g = g

                # perform gradient updates
                (theta_opt, phi_opt) = optimizers
                theta_updates, theta_opt_state = theta_opt.update(
                    theta_g, theta_opt_state, theta)

                # freeze during burn-in
                theta_updates = lax.cond(burn_in, tree_zeros_like,
                                         _identity, theta_updates)

                # perform gradient updates
                theta = optax.apply_updates(theta, theta_updates)
                phi_n_updates, phi_n_opt_states = vmap(phi_opt.update)(
                    phi_n_g, phi_n_opt_states, phi_n)
                phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
                return nvlb, s, theta, phi_n, theta_opt_state, phi_n_opt_states
            return gp_training_step
        else:
            @jit
            def tp_training_step(key, theta, phi_n, theta_opt_state,
                              phi_n_opt_states, x, burn_in):
                (nvlb, s), g = value_and_grad(avg_neg_tp_elbo, argnums=(1, 2),
                                         has_aux=True)(key, theta, phi_n, logpx,
                                                       kernel_fn, x, t, nsamples)
                theta_g, phi_n_g = g

                # perform gradient updates
                (theta_opt, phi_opt) = optimizers
                theta_updates, theta_opt_state = theta_opt.update(
                    theta_g, theta_opt_state, theta)

                # freeze tau if desired
                tau_updates = lax.cond(fix_df, tree_zeros_like,
                                       _identity, theta_updates[2])
                theta_updates = (theta_updates[0], theta_updates[1],
                                 tau_updates)

                # freeze during burn-in
                theta_updates = lax.cond(burn_in, tree_zeros_like,
                                         _identity, theta_updates)

                # perform gradient updates
                theta = optax.apply_updates(theta, theta_updates)
                phi_n_updates, phi_n_opt_states = vmap(phi_opt.update)(
                    phi_n_g, phi_n_opt_states, phi_n)
                phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
                return nvlb, s, theta, phi_n, theta_opt_state, phi_n_opt_states
            return tp_training_step


    ## define evaluation step
    def make_eval_step(logpx, kernel_fn, t, nsamples, elbo_fn):
        @jit
        def eval_step(key, theta, phi_n, x):
            (nvlb, s)  = elbo_fn(key, theta, phi_n, logpx, kernel_fn,
                                 x, t, nsamples)
            return nvlb, s
        return eval_step


    # initialize eval/training step
    if args.eval_only:
        if args.GP:
            elbo_fn = avg_neg_gp_elbo
        else:
            elbo_fn = avg_neg_tp_elbo
        eval_step = make_eval_step(nica_logpx, kernel_fn, t,
                                       nsamples, elbo_fn)
    else:
        training_step = make_training_step(
            nica_logpx, kernel_fn, t ,nsamples,
            (theta_optimizer, phi_optimizer), args.GP, fix_df
        )

    # set up training
    train_data = x.copy()
    if args.resume_ckpt:
        start_epoch = ckpt_epoch+1
        num_epochs = start_epoch+num_epochs
        if args.eval_only:
            start_epoch = 0
            num_epochs = 1
    else:
        start_epoch = 0
        elbo_hist = []
    best_elbo = -jnp.inf
    # train for multiple epochs
    for epoch in range(start_epoch, num_epochs):
        tic = time.perf_counter()
        shuffle_idx, key = rngcall(jr.permutation, key, n_data)
        shuff_data = train_data[shuffle_idx]
        elbo_epoch_hist = []
        burn_in = epoch < args.burn_in_len
        # iterate over all minibatches
        s_samples_all = []
        for it in range(num_minibs):
            x_it = shuff_data[it*minib_size:(it+1)*minib_size]
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


            elbo_epoch_hist.append(-nvlb.item())
            print("*Epoch: [{0}/{1}]\t"
                  "Minibatch: [{2}/{3}]\t"
                  "ELBO: {4:.2f}".format(epoch, num_epochs-1, it,
                                        num_minibs-1, -nvlb))

            toc = time.perf_counter()

            # saving sample
            s_sample = s_sample.swapaxes(-1, -2)
            if args.GP:
                s_sample = s_sample.mean(axis=(1,))
            else:
                s_sample = s_sample.mean(axis=(1, 2))
            s_samples_all.append(s_sample)


        s_samples = jnp.vstack(s_samples_all)
        epoch_avg_elbo = jnp.mean(jnp.array(elbo_epoch_hist))
        print("Epoch [{0}/{1}] took: {2:.2f}\t"
              "AVG. ELBO: {3:.2f}\t"
              "data seed: {4}\t"
              "est. seed: {5}\t"
              "init. theta lr: {6}\t"
              "init. phi lr: {7}".format(epoch, num_epochs-1,
                                         toc-tic, epoch_avg_elbo,
                                         args.data_seed, args.est_seed,
                                         theta_lr, phi_lr))

        # save checkpoints
        if not args.eval_only:
            elbo_hist.append(epoch_avg_elbo.item())
            if epoch_avg_elbo > best_elbo:
                print("**Saving checkpoint (best elbo thus far)**")
                best_elbo = epoch_avg_elbo
                save_checkpoint((epoch, key, theta, phi, theta_opt_state,
                                 phi_opt_states), (elbo_hist), args)

        # plot training histories
        if epoch % args.plot_freq == 0 or args.eval_only:
            plt.plot(elbo_hist)
            plt.tight_layout()
            if args.headless:
                plt.savefig("INFER_cv4a_elbo_hist.png")
            else:
                plt.show(block=False)
                plt.pause(5.)
            plt.close()
    return elbo_hist, s_samples, shuffle_idx



def train_phi(x, t, mean_fn, kernel_fn, args, key):
    # unpack useful args
    N = args.N
    M = x.shape[1]
    L = args.L_est
    T = t.shape[0]
    n_data = x.shape[0]
    n_pseudo = args.num_pseudo
    minib_size = args.minib_size
    minib_size = 8 ### TEST ###
    num_epochs = 1000
    #phi_lr = args.phi_learning_rate
    phi_lr = 3e-2
    if args.GP:
#        nsamples = args.num_s_samples
        nsamples = 25
    else:
#        nsamples = (args.num_s_samples, args.num_tau_samples)
        nsamples = (5, 5)

    # initialize variational parameters (phi) with pseudo-points (tu)
    tu = kmeans2(t, k=n_pseudo, minit='points')[0]
    tu = jnp.tile(tu, (n_data, 1, 1))

    W, key = rngcall(
        lambda _k: vmap(lambda _: jnp.linalg.cholesky(
            jnp.eye(N)
            #sample_wishart(_, jnp.array(N+1.), jnp.eye(N))
        )[jnp.tril_indices(N)], out_axes=-1)(jr.split(_k, n_pseudo)), key
    )

    phi_s = (jnp.repeat(W[None, :], n_data, 0),
             jnp.ones(shape=(n_data, N, n_pseudo)), tu)
    if args.GP:
        phi = phi_s
    else:
        phi_df, key = rngcall(lambda _: vmap(lambda _k: rdm_df(
            _k, min_val=args.tp_df, max_val=args.tp_df))(jr.split(_, n_data*N)), key)
        phi_df = jnp.log(phi_df.reshape(n_data, N))
        phi_tau = (phi_df, phi_df)
        phi = (phi_s, phi_tau)

    # optionally load from checkpoint
    ckpt, hist = load_checkpoint(args)
    _, _, theta, _, _, _ = ckpt
    elbo_hist = hist
    ## set up training details
    num_full_minibs, remainder = divmod(n_data, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    phi_optimizer = optax.adam(phi_lr)
    # note that optimizer states are re-initialized here rather than
    # continued from the ckpt due to completly new data used here.
    # previous phi values are just used for initialization (could also re-init)
    phi_opt_states = vmap(phi_optimizer.init)(phi)


    # define training step
    def make_inference_step(logpx, kernel_fn, t, nsamples, theta, phi_opt, is_gp):
        if is_gp:
            @jit
            def gp_training_step(key, phi_n, phi_n_opt_states, x):
                (nvlb, s), g = value_and_grad(avg_neg_gp_elbo, argnums=2,
                                         has_aux=True)(key, theta, phi_n, logpx,
                                                       kernel_fn, x, t, nsamples)
                phi_n_g = g

                # perform gradient updates
                phi_n_updates, phi_n_opt_states = vmap(phi_opt.update)(
                    phi_n_g, phi_n_opt_states, phi_n)
                phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
                return nvlb, s, phi_n, phi_n_opt_states
            return gp_training_step
        else:
            @jit
            def tp_training_step(key, phi_n, phi_n_opt_states, x):
                (nvlb, s), g = value_and_grad(avg_neg_tp_elbo, argnums=2,
                                         has_aux=True)(key, theta, phi_n, logpx,
                                                       kernel_fn, x, t, nsamples)
                phi_n_g = g

                # perform gradient updates
                phi_n_updates, phi_n_opt_states = vmap(phi_opt.update)(
                    phi_n_g, phi_n_opt_states, phi_n)
                phi_n = vmap(optax.apply_updates)(phi_n, phi_n_updates)
                return nvlb, s, phi_n, phi_n_opt_states
            return tp_training_step


    # initialize eval/training step
    phi_training_step = make_inference_step(
            nica_logpx, kernel_fn, t, nsamples, theta, phi_optimizer, args.GP,
        )

    # set up training
    train_data = x.copy()
    start_epoch = 0
    elbo_hist = []
    best_elbo = -jnp.inf
    # train for multiple epochs
    for epoch in range(start_epoch, num_epochs):
        tic = time.perf_counter()
        elbo_epoch_hist = []
        burn_in = epoch < args.burn_in_len
        # iterate over all minibatches
        s_samples_all = []
        for it in range(num_minibs):
            x_it = train_data[it*minib_size:(it+1)*minib_size]
            # select variational parameters of the observations in minibatch
            idx_set_it = jnp.arange(n_data)[it*minib_size:(it+1)*minib_size]
            phi_it = tree_get_idx(phi, idx_set_it)
            phi_opt_states_it = tree_get_idx(phi_opt_states, idx_set_it)

            # training step (or evaluation)
            (nvlb, s_sample, phi_it, phi_opt_states_it), key = rngcall(
                phi_training_step, key, phi_it, phi_opt_states_it, x_it)

            # update the full variational parameter pytree at right indices
            phi = tree_map(lambda a, b: a.at[idx_set_it].set(b), phi, phi_it)
            phi_opt_states = tree_map(lambda a, b: a.at[idx_set_it].set(b),
                                      phi_opt_states, phi_opt_states_it)

            elbo_epoch_hist.append(-nvlb.item())
            print("*Inference epoch: [{0}/{1}]\t"
                  "Minibatch: [{2}/{3}]\t"
                  "ELBO: {4:.2f}".format(epoch, num_epochs-1, it, num_minibs-1,
                                         -nvlb))

            # compute average sample
            s_sample = s_sample.swapaxes(-1, -2)
            if args.GP:
                s_sample = s_sample.mean(axis=(1,))
            else:
                s_sample = s_sample.mean(axis=(1, 2))
            s_samples_all.append(s_sample)

            toc = time.perf_counter()

        epoch_avg_elbo = jnp.mean(jnp.array(elbo_epoch_hist))
        s_samples = jnp.vstack(s_samples_all)
        print("Inference epoch [{0}/{1}] took: {2:.2f}\t"
              "AVG. ELBO: {3:.2f}\t"
              "data seed: {4}\t"
              "est. seed: {5}\t"
              "init. phi lr: {6}".format(epoch, num_epochs-1,
                                         toc-tic, epoch_avg_elbo,
                                         args.data_seed, args.est_seed, phi_lr))

        # save checkpoints
        elbo_hist.append(epoch_avg_elbo.item())
        if epoch_avg_elbo > best_elbo:
            print("**Saving checkpoint (best elbo thus far)**")
            best_elbo = epoch_avg_elbo
            save_checkpoint((phi, s_samples), (elbo_hist), args, True)

        # plot training histories
        if epoch % args.plot_freq == 0 or args.eval_only:
            plt.plot(elbo_hist)
            plt.tight_layout()
            if args.headless:
                plt.savefig("cv4a_elbo_hist.png")
            else:
                plt.show(block=False)
                plt.pause(5.)
            plt.close()
    return elbo_hist, s_samples
