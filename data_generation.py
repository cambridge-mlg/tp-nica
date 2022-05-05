import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

import pdb

from jax import jit, vmap
from nn import init_nica_params, nica_mlp
from utils import gaussian_sample_from_mu_prec, invmp
from tprocess.sampling import sample_multiple_tprocesses
from tprocess.util import zero_mean_fn
from tprocess.kernels import (
    gen_rdm_Gamma_params,
    gen_rdm_SE_kernel_params,
    SE_kernel
)

import matplotlib.pyplot as plt
import seaborn as sns


def sample_tpnica(key, x, gp_mu_fn_list, gp_k_fn_list, gamma_params_list,
                  mixer_params):
    # sample each IC as a t-process
    key, s_key = jr.split(key)
    s = sample_multiple_tprocesses(s_key, x, gp_mu_fn_list, gp_k_fn_list,
                                   gamma_params_list)
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    return z, s


def gen_tprocess_nica_data(key, x, N, M, L, num_samples, noise_factor=0.15,
                           repeat_layers=False):
    # set-up Gamma prior and GP parameters (used for all samples)
    key, *Gamma_keys = jr.split(key, N+1)
    key, *k_keys = jr.split(key, N+1)
    gamma_params = [gen_rdm_Gamma_params(Gamma_keys[n]) for n in range(N)]
    k_params = [gen_rdm_SE_kernel_params(k_keys[n], x) for n in range(N)]
    k_funcs = [SE_kernel(p) for p in k_params]
    mu_funcs = [zero_mean_fn for _ in range(N)]
    # initialize mixing function parameters
    key, mlp_key = jr.split(key, 2)
    mixer_params = init_nica_params(N, M, L, mlp_key, repeat_layers)

    # sample ICs and mix them
    key, *sample_keys = jr.split(key, num_samples+1)
    z, s = vmap(
        lambda _: sample_tpnica(_, x, mu_funcs, k_funcs,
                                gamma_params, mixer_params)
    )(jnp.vstack(sample_keys))

    # standardize each dim independently so can add apropriate output noise
    z = (z-z.mean(axis=(0, 2), keepdims=True)) / z.std(axis=(0, 2),
                                                       keepdims=True)
    y = z+jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    R = noise_factor*jnp.eye(M)
    return y, z, s, R, mixer_params, k_params, gamma_params

 
if __name__ == "__main__":
    N = 5
    M = 5
    D = 10
    L = 2

    # some locations
    x = jnp.linspace(0, 3, 100)[:, None]
    X, Y = jnp.meshgrid(jnp.linspace(0, 3, 20), jnp.linspace(0, 3, 20))
    x2 = jnp.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # define mean, kernel and paramter dictionary


    # use to plot 2D data
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, s[0, :].reshape(20, 20),
                    rstride=1, cstride=1, cmap='viridis')
    plt.show()

    pdb.set_trace()


    # for 1D data
    #for i in range(N):
    #    sns.lineplot(x.squeeze(), s.T[:,i])
    #plt.show()

    #from scipy.interpolate import interp1d
    #x_int = jnp.linspace(0, 3, 1000)
    #for i in range(N):
    #    y_int = interp1d(x.squeeze(), s.T[:, i], kind='cubic')(x_int)
    #    sns.lineplot(x_int, y_int)
    #plt.show()
















