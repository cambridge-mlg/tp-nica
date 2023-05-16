import os
#os.environ["MPLCONFIGDIR"] = "/proj/herhal/.cache/"

import matplotlib
from matplotlib import projections
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import pdb
import sys

from cv4a_data import get_cv4a_data
#from cv4a_test import classification_test
from jax.config import config
config.update("jax_enable_x64", True)
from jax.dlpack import to_dlpack
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

from train_cv4a import train, train_phi
from data_generation import (
    gen_tpnica_data,
    gen_gpnica_data,
    gen_1d_locations,
    gen_2d_locations
)
from kernels import se_kernel_fn


def parse():
    """Parse args.
    """
    # synthetic data generation args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-N', type=int, default=4,
                        help="number of ICs")
    parser.add_argument('--num-pseudo', type=int, default=50,
                        help="number of pseudo latent points to use")
    parser.add_argument('--L-est', type=int, default=2,
                        help="model: number of nonlinear layers; 0 = linear ICA")
    parser.add_argument('--mean-function', type=str, default="zero",
                        help="zero (zero mean assumed),")
    parser.add_argument('--kernel', type=str, default="se",
                        help="se (squared exponential),")
    parser.add_argument('--tp-df', type=float, default=4.01,
                        help="initialization of df tprocess")
    parser.add_argument('--fix-df', action='store_true', default=False,
                        help="fix df at init value for whole training")
    parser.add_argument('--GP', action='store_true', default=False,
                        help="generate and train from GP latents instead of TP")
    # inference, training and optimization args
    parser.add_argument('--num-s-samples', type=int, default=3,
                        help="num. of samples from q(s|tau) in elbo")
    parser.add_argument('--num-tau-samples', type=int, default=3,
                        help="num. of samples from q(tau) in elbo")
    parser.add_argument('--phi-learning-rate', type=float, default=0.28,
                        help="learning rate for variational params")
    parser.add_argument('--theta-learning-rate', type=float, default=0.004,
                        help="learning rate for model params")
    parser.add_argument('--minib-size', type=int, default=8,
                        help="minibatch size")
    parser.add_argument('--num-epochs', type=int, default=10000,
                        help="number of training epochs")
    parser.add_argument('--num-epochs-infer', type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument('--burn-in-len', type=int, default=0,
                        help="number of epochs to keep theta params fixed")
    # set all ICs to have same distribs
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
    parser.add_argument('--cv4a-dir', type=str, default="cv4a_data/",
                        help="location where data is saved")
    parser.add_argument('--resume-ckpt', action='store_true', default=False,
                        help="resume training if checkpoint for matching\
                        settings exists")
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help="evaluate only, from checkpoint, no training")
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

#    assert args.minib_size <= args.num_data

    # create folder to save checkpoints    
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # ensure there checkpoint will be loaded if only evaluating
    if args.eval_only:
        assert args.resume_ckpt, "'Eval only' requires --resume-ckpt=True"

    # set up observed data
    T_t = 6
    x, areas, field_masks, labels, dates = get_cv4a_data(args.cv4a_dir)
    x = jnp.swapaxes(x, 1, 2)
    x_tr_orig = x[:, :, :T_t, :, :]
    x_te_orig = x[:, :, T_t:2*T_t, :, :]
    num_data, M, _T_t, T_x, T_y = x_tr_orig.shape
    assert _T_t == T_t
    x_tr = x_tr_orig.reshape(num_data, M, -1)
    x_te = x_te_orig.reshape(num_data, M, -1)

    # set up input locations
    t = gen_2d_locations(T_x*T_y)[:, [1, 0]] # this doesnt actually matter
    t = (t-t.mean())/t.std()
    dates = jnp.array([(dates[i] - dates[0]).days for i in
                          range(len(dates))])
    dates = (dates-dates.mean())/dates.std()
    dates_tr = dates[:T_t]
    dates_te = dates[T_t:2*T_t]

    t_tr = jnp.hstack((jnp.repeat(dates_tr, T_x*T_y)[:, None],
                       jnp.tile(t, (T_t, 1))))
    t_te = jnp.hstack((jnp.repeat(dates_te, T_x*T_y)[:, None],
                       jnp.tile(t, (T_t, 1))))

    # train
    #if not args.eval_only:
    #    elbo_hist = train(x_tr, t_tr, mu_fn, k_fn, args, est_key)
    ## perform feature extraction
    #else:
    #    key, infer_key = jr.split(est_key)
    #    elbo_hist, s_features = train_phi(x_te, t_te, mu_fn, k_fn, args, infer_key)

    ## test features
    #s_features = s_features.reshape(num_data, args.N, T_t, T_x, T_y)
    #out = classification_test(to_dlpack(s_features, True), labels,
    #                          field_masks, True)
    out = classification_test(to_dlpack(x_tr_orig, True), labels, field_masks,
                                  False)


if __name__=="__main__":
    #with jax.debug_nans():
    sys.exit(main())
