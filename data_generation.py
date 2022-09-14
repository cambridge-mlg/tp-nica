from jax.config import config
config.update("jax_enable_x64", True)


import jax
import jax.numpy as jnp
import jax.random as jr


import pdb

from jax import vmap, lax
from nn import init_nica_params, nica_mlp
from kernels import rdm_df, rdm_SE_kernel_params, compute_K, se_kernel_fn

import matplotlib.pyplot as plt


def gen_1d_locations(T):
    return jnp.arange(T, dtype=jnp.float64)[:, None]


def gen_2d_locations(T):
    t1, t2 = jnp.meshgrid(jnp.arange(int(jnp.sqrt(T))),
                        jnp.arange(int(jnp.sqrt(T))))
    return jnp.hstack((t1.flatten()[:, None], t2.flatten()[:, None]))


def sample_tprocess(key, latent_inputs, gp_mu_fn, gp_kernel_fn,
                    gp_kernel_params, df):
    # sample tau from Gamma prior
    tau_key, tp_key = jr.split(key)
    tau_sample = jr.gamma(tau_key, df/2, shape=())/(df/2)
    # define GP parameters
    mu = vmap(gp_mu_fn)(latent_inputs)
    K = compute_K(latent_inputs, gp_kernel_fn, gp_kernel_params)
    # sample
    tp_sample = jr.multivariate_normal(tp_key, mu, (1/tau_sample)*K)
    return tp_sample, tau_sample


def sample_tpnica(key, t, gp_mu_fn, gp_k_fn, gp_k_params, df, mixer_params):
    # sample each IC as a t-process
    N = df.shape[0]
    key, *s_key = jr.split(key, N+1)
    s, tau = vmap(
        lambda _a, _b, _c: sample_tprocess(_a, t, gp_mu_fn, gp_k_fn, _b, _c)
    )(jnp.vstack(s_key), gp_k_params, df)
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    return z, s, tau


def gen_tprocess_nica_data(key, t, N, M, L, num_samples, mu_func, kernel_func,
                           noise_factor=0.15, repeat_layers=False,
                           repeat_dfs=False, repeat_kernels=False):
    # set-up Gamma prior and GP parameters (used for all samples)
    key, *gamma_keys = jr.split(key, N+1)
    key, *k_keys = jr.split(key, N+1)
    if repeat_dfs:
        gamma_keys = [gamma_keys[0]]*len(gamma_keys)
    if repeat_kernels:
        k_keys = [k_keys[0]]*len(k_keys)
    dfs = vmap(rdm_df)(jnp.vstack(gamma_keys))
    k_params = vmap(lambda _: rdm_SE_kernel_params(_))(jnp.vstack(k_keys))

    # initialize mixing function parameters
    key, mlp_key = jr.split(key, 2)
    mixer_params = init_nica_params(mlp_key, N, M, L, repeat_layers)

    # sample ICs and their mixtures
    key, *sample_keys = jr.split(key, num_samples+1)
    z, s, tau = lax.map(
        lambda _: sample_tpnica(_, t, mu_func, kernel_func, k_params,
                                dfs, mixer_params), jnp.vstack(sample_keys))

    # standardize each dim independently so can add apropriate output noise
    z = (z-z.mean(axis=(0, 2), keepdims=True)) / z.std(axis=(0, 2),
                                                       keepdims=True)
    x = z+jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    Q = noise_factor*jnp.eye(M)
    return x, z, s, tau, Q, mixer_params, k_params, dfs


if __name__ == "__main__":
    N = 5
    M = 5
    D = 10
    L = 2
    T = 1000

    # some locations
    t = gen_1d_locations(T)
    mu_fn = lambda _: 0
    cov_fn = se_kernel_fn
    rng = jr.PRNGKey(0)
    df = jnp.array(1000.)
    rng, rng0 = jr.split(rng)
    theta_cov = rdm_SE_kernel_params(rng0)
    sample, tau = vmap(lambda _: sample_tprocess(_, t, mu_fn, cov_fn,
            theta_cov, df))(jr.split(rng, 10))
    plt.plot(sample.T)
    plt.show()



    # use to plot 2D data
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, s[0, :].reshape(20, 20),
    #                rstride=1, cstride=1, cmap='viridis')
    #plt.show()

    #pdb.set_trace()


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

