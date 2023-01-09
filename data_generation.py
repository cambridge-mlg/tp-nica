from jax.config import config
config.update("jax_enable_x64", True)


import jax
import jax.numpy as jnp
import jax.random as jr

import pdb

from jax import vmap, lax, jit
from nn import init_nica_params, nica_mlp
from kernels import rdm_df, rdm_SE_kernel_params, compute_K, se_kernel_fn
from functools import partial

import matplotlib.pyplot as plt


def gen_1d_locations(T):
    return jnp.arange(T, dtype=jnp.float64)[:, None]


def gen_2d_locations(T):
    t1, t2 = jnp.meshgrid(jnp.arange(int(jnp.sqrt(T)), dtype=jnp.float64),
                        jnp.arange(int(jnp.sqrt(T)), dtype=jnp.float64))
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


def sample_gp(key, latent_inputs, gp_mu_fn, gp_kernel_fn, gp_kernel_params):
    # define GP parameters
    mu = vmap(gp_mu_fn)(latent_inputs)
    K = compute_K(latent_inputs, gp_kernel_fn, gp_kernel_params)
    return jr.multivariate_normal(key, mu, K)


def sample_tpnica(key, t, gp_mu_fn, gp_k_fn, gp_k_params, df, mixer_params):
    # sample each IC as a t-process
    N = df.shape[0]
    key, *s_key = jr.split(key, N+1)
    sample_fun = lambda a, b, c: sample_tprocess(a, t, gp_mu_fn, gp_k_fn, b, c)
    s, tau = vmap(sample_fun)(jnp.vstack(s_key), gp_k_params, df)
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    return z, s, tau


def sample_gpnica(key, t, gp_mu_fn, gp_k_fn, gp_k_params, mixer_params):
    # sample each IC as a GP
    N = mixer_params[0].shape[0]
    key, *s_key = jr.split(key, N+1)
    sample_fun = lambda _a, _b: sample_gp(_a, t, gp_mu_fn, gp_k_fn, _b)
    s = vmap(sample_fun)(jnp.vstack(s_key), gp_k_params)
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    return z, s


@partial(jit, static_argnames=( "N", "M", "L", "num_samples", "mu_func",
    "kernel_func", "repeat_dfs", "repeat_kernels"))
def gen_tpnica_data(key, t, N, M, L, num_samples, mu_func, kernel_func,
                    tp_df=2.01, noise_factor=0.15, repeat_layers=False,
                    repeat_dfs=False, repeat_kernels=False):
    # set-up Gamma prior and GP parameters (used for all samples)
    D = t.shape[-1]
    key, *gamma_keys = jr.split(key, N+1)
    key, *k_keys = jr.split(key, N+1)
    if repeat_dfs:
        dfs = jnp.ones((N,))*tp_df
    else:
        dfs = vmap(rdm_df)(jnp.vstack(gamma_keys))
    if repeat_kernels:
        k_keys = [k_keys[0]]*len(k_keys)
    if D == 1:
        k_params = vmap(rdm_SE_kernel_params)(jnp.vstack(k_keys))
    elif D == 2:
        k_params = vmap(rdm_SE_kernel_params, in_axes=(0, None, None))(
            jnp.vstack(k_keys), 4., 10.)

    # initialize mixing function parameters
    key, mlp_key = jr.split(key, 2)
    mixer_params = init_nica_params(mlp_key, N, M, L, repeat_layers)

    # sample ICs and their mixtures
    key, *sample_keys = jr.split(key, num_samples+1)
    sample_fun = lambda _: sample_tpnica(_, t, mu_func, kernel_func, k_params,
            dfs, mixer_params)
    z, s, tau = lax.map(sample_fun, jnp.vstack(sample_keys))

    # add noise
    #x = z + jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    #Q = jnp.eye(M)*noise_factor
    z = (z-z.mean(axis=(0, 2), keepdims=True)) / z.std(
        axis=(2,), keepdims=True).mean(0, keepdims=True)
    x = z+jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    Q = noise_factor*jnp.eye(M)
    return x, z, s, tau, Q, mixer_params, k_params, dfs


@partial(jit, static_argnames=( "N", "M", "L", "num_samples", "mu_func",
                               "kernel_func", "repeat_kernels"))
def gen_gpnica_data(key, t, N, M, L, num_samples, mu_func, kernel_func,
                    noise_factor=0.15, repeat_layers=False,
                    repeat_kernels=False):
    D = t.shape[-1]
    # set-up GP parameters (used for all samples)
    key, *k_keys = jr.split(key, N+1)
    if repeat_kernels:
        k_keys = [k_keys[0]]*len(k_keys)
    if D == 1:
        k_params = vmap(rdm_SE_kernel_params)(jnp.vstack(k_keys))
    elif D == 2:
        k_params = vmap(rdm_SE_kernel_params, in_axes=(0, None, None))(
            jnp.vstack(k_keys), 4., 10.)

    # initialize mixing function parameters
    key, mlp_key = jr.split(key, 2)
    mixer_params = init_nica_params(mlp_key, N, M, L, repeat_layers)

    # sample ICs and their mixtures
    key, *sample_keys = jr.split(key, num_samples+1)
    sample_fun = lambda _: sample_gpnica(_, t, mu_func, kernel_func, k_params,
                                         mixer_params)
    z, s = lax.map(sample_fun, jnp.vstack(sample_keys))

    # add noise
    #x = z + jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    #Q = jnp.eye(M)*noise_factor
    z = (z-z.mean(axis=(0, 2), keepdims=True)) / z.std(
        axis=(2,), keepdims=True).mean(0, keepdims=True)
    x = z+jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    Q = noise_factor*jnp.eye(M)
    return x, z, s, Q, mixer_params, k_params



if __name__ == "__main__":
    N = 5
    M = 5
    D = 100
    L = 2
    T = 1000

    # some locations
    t = gen_1d_locations(T)
    mu_fn = lambda _: 0
    cov_fn = se_kernel_fn
    rng = jr.PRNGKey(0)
    df = jnp.array(2.)
    rng, rng0 = jr.split(rng)
    theta_cov = rdm_SE_kernel_params(rng0)
    tp_sample, tau = vmap(lambda _: sample_tprocess(_, t, mu_fn, cov_fn,
            theta_cov, df))(jr.split(rng, D))
    plt.plot(tp_sample.T)

    gp_sample = vmap(lambda _: sample_gp(_, t, mu_fn, cov_fn,
            theta_cov))(jr.split(rng, D))
    plt.plot(gp_sample.T, 'b--')
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

