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
from cv4a_test import test_rf
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
    parser.add_argument('-N', type=int, default=6,
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
    parser.add_argument('--num-s-samples', type=int, default=5,
                        help="num. of samples from q(s|tau) in elbo")
    parser.add_argument('--num-tau-samples', type=int, default=5,
                        help="num. of samples from q(tau) in elbo")
    parser.add_argument('--phi-learning-rate', type=float, default=3e-2,
                        help="learning rate for variational params")
    parser.add_argument('--theta-learning-rate', type=float, default=3e-4,
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
    x, areas, fields, field_masks, labels, dates = get_cv4a_data(args.cv4a_dir)
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
    #    elbo_hist, s_features, shuff_idx = train(x_tr, t_tr, mu_fn,
    #                                             k_fn, args, est_key)
    ## perform feature extraction
    #else:
    #    key, infer_key = jr.split(est_key)
    #    elbo_hist, s_features = train_phi(x_te, t_te, mu_fn, k_fn, args, infer_key)



    #def get_field_pxs(x, field_masks, labels, fields, dim, T):
    #    x2 = x*field_masks[:, None, :, :, :]
    #    pxs = []
    #    pxs_labs = []
    #    pxs_fields = []
    #    for i in range(x2.shape[0]):
    #        print(i)
    #        px = x2[i].reshape(dim, T, -1)
    #        px = np.moveaxis(px, -1, 0)
    #        idx = np.argwhere(px.sum((1, 2)) != 0)
    #        pxs.append(px[idx])
    #        _lab = labels[i]
    #        _fld = fields[i]
    #        pxs_labs.extend([_lab]*len(idx))
    #        pxs_fields.extend([_fld]*len(idx))
    #    px_x = jnp.vstack(pxs).squeeze()
    #    px_y = jnp.array(pxs_labs)
    #    px_id = jnp.array(pxs_fields)
    #    return px_x, px_y, px_id



    ### test features
    #nd = 1000
    #s_features = s_features.reshape(num_data, args.N, T_t, T_x, T_y)
    #x_use = s_features
    ##x_use = x_te_orig
    #x_use = x_use[:nd]
    #f_masks = field_masks[:nd]
    #lab_use = labels[:nd]
    #f_use = fields[:nd]

    #px_x, px_y, px_id = get_field_pxs(x_use, f_masks, lab_use, f_use, M, T_t)
    #px_x = px_x.reshape(px_x.shape[0], -1)
###############################

    #s_features = s_features.reshape(num_data, args.N, T_t, T_x, T_y)
    #sf_use = s_features
    ##sf_use = x_te_orig
    #sf_use = jr.normal(jr.PRNGKey(args.test_seed), (num_data, args.N,
    #                                                T_t, T_x, T_y))
    #sf = sf_use.swapaxes(1, 2).reshape(-1, args.N, T_x, T_y)
    #time_classes = jnp.tile(jnp.arange(T_t), num_data)

    ##n_ic = 0
    ##sf = sf[:, n_ic, :, :]
    #sf = sf.reshape(sf.shape[0], -1)
    #pdb.set_trace()
    #losses, accs = test_rf(sf, time_classes)

if __name__=="__main__":
    #with jax.debug_nans():
    sys.exit(main())
