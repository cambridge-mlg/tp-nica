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


def gen_tprocess_nica_data(x, N, M, L, num_samples, noise_factor=0.39,
                           repeat_layers=False):
    """note noise_factor is in terms of std, so if nf=0.39-> noisy data has
    approx. 1.15 times higher variance.
    """
    def _sample(key):
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

        pdb.set_trace()

        # standardize each dim independently so can add apropriate output noise
        z = (z-z.mean(axis=(0, 2), keepdims=True)) / z.std(axis=(0, 2),
                                                           keepdims=True)
        y = z+noise_factor*jr.normal(key, shape=z.shape)
        return 0
    return _sample

    # add output noise (scaled to be appropriate size)
    #M = z.shape[0]
    #z_vars = jnp.diag(jnp.cov(z))
    #key, noise_key = jr.split(key)
    #noise_vars = jr.uniform(noise_key, shape=(M,), minval=0.05*z_vars,
    #                        maxval=0.2*z_vars)
    ## get precision matrix from cov matrix
    #R = invmp(noise_vars*jnp.eye(M), jnp.eye(M))
    #y_keys = jr.split(key, z.shape[1])
    #y = vmap(gaussian_sample_from_mu_prec, (1, None, 0), 1)(z, R, y_keys)
 
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
    key = jr.PRNGKey(0)
    keys = jr.split(key, D)
    sampler = gen_tprocess_nica_data(x2, N, M, L)
    y, z, s, params = sampler(key)


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
















