import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

import pdb

from jax import jit, vmap
from nn import init_nica_params, nica_mlp
from tprocess.sampling import sample_tprocess
from tprocess.util import zero_mean_fn
from tprocess.kernels import (
    rdm_df,
    rdm_SE_kernel_params,
    se_kernel_fn
)

import matplotlib.pyplot as plt
import seaborn as sns


def sample_tpnica(key, t, gp_mu_fn, gp_k_fn, gp_k_params, df, mixer_params):
    # sample each IC as a t-process
    N = df.shape[0]
    key, *s_key = jr.split(key, N+1)
    s, r = vmap(
        lambda _a, _b, _c: sample_tprocess(_a, t, gp_mu_fn, gp_k_fn, _b, _c)
    )(jnp.vstack(s_key), gp_k_params, df)
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    return z, s, r


def gen_tprocess_nica_data(key, t, N, M, L, num_samples,
                           mu_func=zero_mean_fn, kernel_func=se_kernel_fn,
                           noise_factor=0.15, repeat_layers=False):
    # set-up Gamma prior and GP parameters (used for all samples)
    key, *gamma_keys = jr.split(key, N+1)
    key, *k_keys = jr.split(key, N+1)
    dfs = vmap(rdm_df)(jnp.vstack(gamma_keys))
    k_params = vmap(lambda _: rdm_SE_kernel_params(_, t))(jnp.vstack(k_keys))

    # initialize mixing function parameters
    key, mlp_key = jr.split(key, 2)
    mixer_params = init_nica_params(mlp_key, N, M, L, repeat_layers)

    # sample ICs and their mixtures
    key, *sample_keys = jr.split(key, num_samples+1)
    z, s, r = vmap(
        lambda _: sample_tpnica(_, t, mu_func, kernel_func, k_params,
                                dfs, mixer_params)
    )(jnp.vstack(sample_keys))

    # standardize each dim independently so can add apropriate output noise
    z = (z-z.mean(axis=(0, 2), keepdims=True)) / z.std(axis=(0, 2),
                                                       keepdims=True)
    x = z+jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    Q = noise_factor*jnp.eye(M)
    return x, z, s, r, Q, mixer_params, k_params, dfs


if __name__ == "__main__":
    N = 5
    M = 5
    D = 10
    L = 2

    # some locations
    t = jnp.linspace(0, 3, 100)[:, None]
    X, Y = jnp.meshgrid(jnp.linspace(0, 3, 20), jnp.linspace(0, 3, 20))
    t2 = jnp.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # define mean, kernel and paramter dictionary


    # use to plot 2D data
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, s[0, :].reshape(20, 20),
                    rstride=1, cstride=1, cmap='viridis')
    plt.show()

    pdb.set_trace()


    # for 1D data
    #for i in range(N):
    #    sns.lineplot(t.squeeze(), s.T[:,i])
    #plt.show()

    #from scipy.interpolate import interp1d
    #t_int = jnp.linspace(0, 3, 1000)
    #for i in range(N):
    #    y_int = interp1d(t.squeeze(), s.T[:, i], kind='cubic')(t_int)
    #    sns.lineplot(t_int, y_int)
    #plt.show()
