import os
#os.environ["MPLCONFIGDIR"] = "/proj/herhal/.cache/"

import matplotlib
from matplotlib import projections
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import pdb
import sys

from numpy import dtype

from jax.config import config
config.update("jax_enable_x64", True)

from linear_baseline import linearICA_eval
from sklearn.linear_model import LinearRegression as LR

###DEBUG##############################
#config.update('jax_disable_jit', True)
#config.update("jax_debug_nans", True)
######################################

import jax
import jax.random as jr
import jax.numpy as jnp
import seaborn as sns

print(jax.devices())

from train import train
from data_generation import (
    gen_data,
    gen_1d_locations,
    gen_2d_locations
)
from kernels import se_kernel_fn


def parse():
    """Parse args.
    """
    # synthetic data generation args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', type=int, default=6,
                        help="number of ICs")
    parser.add_argument('-M', type=int, default=12,
                        help="dimension of each observed data point")
    parser.add_argument('-T', type=int, default=1024,
                        help="number of latent input locations")
    parser.add_argument('--num-pseudo', type=int, default=50,
                        help="number of pseudo latent points to use")
    parser.add_argument('-D', type=int, default=2,
                        help="dimension of latent input locations")
    parser.add_argument('--num-data', type=int, default=256,
                        help="total number of data samples to generate")
    parser.add_argument('--L-data', type=int, default=0,
                        help="data gen: number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('--L-est', type=int, default=0,
                        help="model: number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('--mean-function', type=str, default="zero",
                        help="zero (zero mean assumed),")
    parser.add_argument('--kernel', type=str, default="se",
                        help="se (squared exponential),")
    parser.add_argument('--tp-df', type=float, default=4.01,
                        help="df of t-process for simulated data")
    parser.add_argument('--GP', action='store_true', default=False,
                        help="generate and train from GP latents instead of TP")
    # inference, training and optimization args
    parser.add_argument('--num-s-samples', type=int, default=4,
                        help="num. of samples from q(s|tau) in elbo")
    parser.add_argument('--num-tau-samples', type=int, default=4,
                        help="num. of samples from q(tau) in elbo")
    parser.add_argument('--phi-learning-rate', type=float, default=0.28,
                        help="learning rate for variational params")
    parser.add_argument('--theta-learning-rate', type=float, default=0.004,
                        help="learning rate for model params")
    parser.add_argument('--minib-size', type=int, default=8,
                        help="minibatch size")
    parser.add_argument('--num-epochs', type=int, default=10000,
                        help="number of training epochs")
    parser.add_argument('--burn-in-len', type=int, default=10,
                        help="number of epochs to keep theta params fixed")
    # set all ICs to have same distribs
    parser.add_argument('--repeat-dfs', action='store_true', default=False,
                        help="force all tprocesses to same degrees of freedom")
    parser.add_argument('--repeat-kernels', action='store_true', default=False,
                        help="force all processes have same kernel params")
    parser.add_argument('--gen-repeat-dfs', action='store_true', default=False,
                        help="generate all processes to same degrees of freedom")
    parser.add_argument('--gen-repeat-kernels', action='store_true', default=False,
                        help="generate all processes to use the same kernel")
    # set seeds
    parser.add_argument('--data-seed', type=int, default=1,
                        help="seed for initializing data generation")
    parser.add_argument('--est-seed', type=int, default=50,
                        help="seed for initializing learning/inference")
    parser.add_argument('--test-seed', type=int, default=99,
                        help="seed for all kinds misc. testing")
    # plotting frequency
    parser.add_argument('--plot-freq', type=int, default=100,
                        help="plot components every n epoch")
    # checkpoint saving, loading, and evaluation
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where data is saved")
    parser.add_argument('--resume-ckpt', action='store_true', default=False,
                        help="resume training if checkpoint for matching\
                        settings exists")
    parser.add_argument('--eval-linear-ica', action='store_true', default=False,
                        help="evaluate linear ICA baseline on data")
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help="evaluate only, from checkpoint, no training")
    # for debugging
    parser.add_argument('--use-gt-nica', action='store_true', default=False,
                        help="set nonlinear ica params to ground-truth values")
    parser.add_argument('--use-gt-kernel', action='store_true', default=False,
                        help="set GP kernel params to ground-truth values")
    parser.add_argument('--use-gt-tau', action='store_true', default=True,
                        help="set tau to ground-truth values")
    # server settings
    parser.add_argument('--headless', action='store_true', default=False,
                        help="switch behaviour on server")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    print("Args: ", args)

    # set prng keys
    data_key = jr.PRNGKey(args.data_seed)
    est_key = jr.PRNGKey(args.est_seed)

    # set mean function
    if args.mean_function == "zero":
        mu_fn = lambda _: 0.
    else:
        raise NotImplementedError
    # and kernel
    if args.kernel == "se":
        k_fn = se_kernel_fn
    else:
        raise NotImplementedError

    assert args.minib_size <= args.num_data
    import time

    # generate synthetic data
    if args.D == 1:
        t = gen_1d_locations(args.T)
    elif args.D == 2:
        assert jnp.sqrt(args.T) % 1 == 0
        t = gen_2d_locations(args.T)


    mean_nrs = 1.0
    noise_factor = 0.1
    while mean_nrs < 1.05 or mean_nrs > 1.15:
        data_key, _ = jr.split(data_key)
        x, z, s, tau, *params = gen_data(data_key, t, args.N, args.M,
                              args.L_data, args.num_data, mu_fn, k_fn,
                              args.tp_df, repeat_kernels=args.gen_repeat_kernels,
                              noise_factor=noise_factor,
                              repeat_dfs=args.gen_repeat_dfs, tp=not args.GP)

        # check that noise is appropriate level
        mean_nrs = jnp.mean(x.var(2) / z.var(2), 0).mean()
        print("Noise-ratio mean.: {0:.2f}".format(mean_nrs))
        if mean_nrs < 1.05:
            noise_factor = 1.5*noise_factor
        elif mean_nrs > 1.15:
            noise_factor = 0.5*noise_factor


    # evaluate by linear ICA
    if args.eval_linear_ica:
        s_lica, lica_mcc = linearICA_eval(x, s)
        print("Linear ICA: {0:.2f}".format(lica_mcc))


    # visualize example in 3D
    if args.D == 2:
        T_sqrt = int(jnp.sqrt(args.T))
        X, Y = jnp.meshgrid(jnp.arange(T_sqrt), jnp.arange(T_sqrt))
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, s[0][0, :].reshape(T_sqrt, T_sqrt),
                        rstride=1, cstride=1, cmap='viridis')
        plt.show()

    # create folder to save checkpoints    
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # ensure there checkpoint will be loaded if only evaluating
    if args.eval_only:
        assert args.resume_ckpt, "Eval only requires --resume-ckpt=True"

    # train model
    elbo_hist, mcc_hist = train(x, z, s, t, mu_fn, k_fn, params, args, est_key)


if __name__=="__main__":
    #with jax.debug_nans():
    sys.exit(main())