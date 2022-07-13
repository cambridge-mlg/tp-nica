import os
os.environ["MPLCONFIGDIR"] = "/proj/herhal/.cache/"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import pdb
import sys

from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as jr
import jax.numpy as jnp
import seaborn as sns

from train import train
from data_generation import (
    gen_tprocess_nica_data,
    gen_1d_locations,
    gen_2d_locations
)
from kernels import se_kernel_fn


def parse():
    """Parse args.
    """
    # synthetic data generation args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', type=int, default=3,
                        help="number of ICs")
    parser.add_argument('-M', type=int, default=3,
                        help="dimension of each observed data point")
    parser.add_argument('-T', type=int, default=200,
                        help="number of latent input locations")
    parser.add_argument('--num-pseudo', type=int, default=50,
                        help="number of pseudo latent points to use")
    parser.add_argument('-D', type=int, default=1,
                        help="dimension of latent input locations")
    parser.add_argument('--num-data', type=int, default=1,
                        help="total number of data samples to generate")
    parser.add_argument('-L', type=int, default=0,
                        help="number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('--mean-function', type=str, default="zero",
                        help="zero (zero mean assumed),")
    parser.add_argument('--kernel', type=str, default="se",
                        help="se (squared exponential),")
    # inference, training and optimization args
    parser.add_argument('--diag-approx', action='store_true', default=False,
                        help="approx. likelih. factor with diagonal Gaussian")
    parser.add_argument('--num-s-samples', type=int, default=5,
                        help="num. of samples from q(s|tau) in elbo")
    parser.add_argument('--num-tau-samples', type=int, default=10,
                        help="num. of samples from q(tau) in elbo")
    parser.add_argument('--learning-rate', type=float, default=3e-2,
                        help="learning rate for training")
    parser.add_argument('--minib-size', type=int, default=1,
                        help="minibatch size")
    parser.add_argument('--num-epochs', type=int, default=10000,
                        help="number of training epochs")
    # set seeds
    parser.add_argument('--data-seed', type=int, default=1,
                        help="seed for initializing data generation")
    parser.add_argument('--est-seed', type=int, default=50,
                        help="seed for initializing learning/inference")
    # plotting frequency
    parser.add_argument('--plot-freq', type=int, default=100,
                        help="plot components every n epoch")
    # saving and loading
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where data is saved")
    # for debugging
    parser.add_argument('--use-gt-nica', action='store_true', default=False,
                        help="set nonlinear ica params to ground-truth values")
    parser.add_argument('--use-gt-kernel', action='store_true', default=False,
                        help="set GP kernel params to ground-truth values")
    parser.add_argument('--use-gt-tau', action='store_true', default=False,
                        help="set tau to ground-truth values")
    # server settings
    parser.add_argument('--headless', action='store_true', default=False,
                        help="switch behaviour on server")
    args = parser.parse_args()
    return args


def main():
    args = parse()

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

    # generate synthetic data
    if args.D == 1:
        t = gen_1d_locations(args.T)
    elif args.D == 2:
        assert jnp.sqrt(args.T) % 1 == 0
        t = gen_2d_locations(args.T)

    x, z, s, tau, *params = gen_tprocess_nica_data(data_key, t, args.N, args.M,
                                                   args.L, args.num_data, mu_fn,
                                                   k_fn)
    # train model
    mcc_hist, elbo_hist = train(x, z, s, t, mu_fn, k_fn, params, args, est_key)
    pdb.set_trace()


if __name__=="__main__":
    sys.exit(main())

