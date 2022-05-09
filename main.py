import argparse
import pdb
import sys

from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from jax import jit, vmap
from train import train
from data_generation import gen_tprocess_nica_data
from tprocess.sampling import gen_1d_locations, gen_2d_locations


def parse():
    """Parse args.
    """
    # synthetic data generation args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', type=int, default=5,
                        help="number of ICs")
    parser.add_argument('-M', type=int, default=5,
                        help="dimension of each observed data point")
    parser.add_argument('-T', type=int, default=100,
                        help="number of latent input locations")
    parser.add_argument('-D', type=int, default=1,
                        help="dimension of latent input locations")
    parser.add_argument('--num-data', type=int, default=10000,
                        help="total number of data samples to generate")
    parser.add_argument('-L', type=int, default=2,
                        help="number of nonlinear layers; 0 = linear ICA")
    # inference, training and optimization args
    parser.add_argument('--inference-iters', type=int, default=5,
                        help="num. of inference iterations")
    parser.add_argument('--num-samples', type=int, default=1,
                        help="num. of samples for elbo")
    parser.add_argument('--learning-rate', type=float, default=1e-2,
                        help="learning rate for training")
    parser.add_argument('--minib-size', type=float, default=32,
                        help="minibatch size")
    parser.add_argument('--num-epochs', type=float, default=100,
                        help="number of training epochs")
    # set seeds
    parser.add_argument('--data-seed', type=int, default=0,
                        help="seed for initializing data generation")
    parser.add_argument('--est-seed', type=int, default=50,
                        help="seed for initializing learning/inference")
    # saving and loading
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where data is saved")

    args = parser.parse_args()
    return args


def main():
    args = parse()

    # set prng keys
    data_key = jr.PRNGKey(args.data_seed)
    est_key = jr.PRNGKey(args.est_seed)

    # generate synthetic data
    if args.D == 1:
        t = gen_1d_locations(args.T)
    elif args.D == 2:
        assert jnp.sqrt(args.T) % 1 == 0
        t = gen_2d_locations(args.T)
    x, z, s, *params = gen_tprocess_nica_data(data_key, t, args.N,
                                              args.M, args.L, args.num_data)

    # train model
    train(x, z, s, t, params, args, est_key)


if __name__=="__main__":
    sys.exit(main())

