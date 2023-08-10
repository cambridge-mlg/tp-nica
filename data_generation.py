from jax.config import config
config.update("jax_enable_x64", True)


import jax
import jax.numpy as jnp
import jax.random as jr

import pdb

from jax import vmap, lax, jit
from jax.nn import sigmoid
from nn import init_nica_params, nica_mlp
from kernels import rdm_df, rdm_SE_kernel_params, compute_K, se_kernel_fn
from functools import partial

import matplotlib.pyplot as plt


def gen_1d_locations(T):
    t = jnp.arange(T, dtype=jnp.float64)[:, None]
    t = (t-t.mean())/t.std()
    return t


def gen_2d_locations(T):
    t1, t2 = jnp.meshgrid(jnp.arange(int(jnp.sqrt(T)), dtype=jnp.float64),
                        jnp.arange(int(jnp.sqrt(T)), dtype=jnp.float64))
    t = jnp.hstack((t1.flatten()[:, None], t2.flatten()[:, None]))
    t = (t-t.mean())/t.std()
    return t


def rational_quad(y_u, y_l, d_u, d_l, eps, s_k):
    rq = (y_u - y_l) * (s_k * eps ** 2 + d_l * eps * (1 - eps))
    rq = y_l + rq / (s_k + (d_u + d_l - 2*s_k)*eps*(1-eps))
    return rq


def fit_rational_quad(x, x_knots, y_knots, deltas):
    u_idx = jnp.argmax(x_knots - x > 0)
    l_idx = u_idx - 1
    x_u = x_knots[u_idx]
    x_l = x_knots[l_idx]
    y_u = y_knots[u_idx]
    y_l = y_knots[l_idx]
    d_u = deltas[u_idx]
    d_l = deltas[l_idx]
    eps = (x - x_l) / (x_u - x_l)
    s_k = (y_u - y_l) / (x_u - x_l)
    rq  = rational_quad(y_u, y_l, d_u, d_l, eps, s_k)
    return rq


def mono_spline(key, x, minval=-1, maxval=1, delta_max=5., num_knots=5):
    internal_knots = jnp.sort(jr.uniform(key, minval=minval,
                              maxval=maxval, shape=(2, num_knots)))
    knots = jnp.hstack((jnp.ones((2, 1)) * minval, internal_knots,
                        jnp.ones((2, 1))*maxval))
    key, key_b = jr.split(key)
    internal_deltas = jr.uniform(key_b, minval=0, maxval=delta_max,
                                  shape=(num_knots,))
    deltas = jnp.hstack((1., internal_deltas, 1.))
    x_knots = knots[0]
    y_knots = knots[1]
    spline = vmap(lambda _: fit_rational_quad(_, x_knots, y_knots, deltas))(x)
    sgn = jr.choice(key, jnp.array([-1., 1.]))
    return sgn * spline


def gen_matrix(key, in_dim, out_dim):
    A = jr.uniform(key, (out_dim, in_dim), minval= -2., maxval=2.)
    cond = jnp.linalg.cond(A)
    return A, cond


def mono_spline_layer(key, x, out_dim):
    key, key_b = jr.split(key)
    y = vmap(mono_spline)(jr.split(key_b, x.shape[0]), x)
    keys = jr.split(key, 1000)
    As, conds = vmap(gen_matrix, (0, None, None))(keys, x.shape[0], out_dim)
    target_cond = jnp.percentile(conds, 25)
    target_idx = jnp.argmin(jnp.abs(conds-target_cond))
    return As[target_idx] @ y


def multilayer_spline(key, x, n_layers, out_dim):
    for _ in range(n_layers):
        x = mono_spline_layer(key, x, out_dim)
        key, _ = jr.split(key)
    return x


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
    sample_fun = lambda _: sample_tprocess(_[0], t, gp_mu_fn, gp_k_fn,
                                           _[1], _[2])
    s, tau = lax.map(sample_fun, (jnp.vstack(s_key), gp_k_params, df))
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    #z = multilayer_spline(key, s, len(mixer_params), mixer_params[0].shape[1])
    return z, s, tau


def sample_gpnica(key, t, gp_mu_fn, gp_k_fn, gp_k_params, mixer_params):
    # sample each IC as a GP
    N = mixer_params[0].shape[0]
    key, *s_key = jr.split(key, N+1)
    sample_fun = lambda _: sample_gp(_[0], t, gp_mu_fn, gp_k_fn, _[1])
    s = lax.map(sample_fun, (jnp.vstack(s_key), gp_k_params))
    # tau belwo is generated just to make it compatible with the tp func above
    _tau = jnp.ones((N,))
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    #z = multilayer_spline(key, s, len(mixer_params), mixer_params[0].shape[1])
    return z, s, _tau


@partial(jit, static_argnames=( "N", "M", "L", "num_samples", "mu_func",
    "kernel_func", "repeat_dfs", "repeat_kernels"))
def gen_data(key, t, N, M, L, num_samples, mu_func, kernel_func,
             tp_df=4.01, noise_factor=0.15, repeat_layers=False,
             repeat_dfs=False, repeat_kernels=False, tp=True):
    # set-up Gamma prior and GP parameters (used for all samples)
    D = t.shape[-1]
    key, *k_keys = jr.split(key, N + 1)
    if tp:
        key, *gamma_keys = jr.split(key, N+1)
        if repeat_dfs:
            dfs = jnp.ones((N,))*tp_df
        else:
            dfs = vmap(lambda _: rdm_df(_, min_val=tp_df, max_val=tp_df+4))(
                jnp.vstack(gamma_keys)
            )
    else:
        # dfs need to be set even for GP because jax requires static shapes...
        dfs = jnp.array([jnp.inf]*N)
    if repeat_kernels:
        k_keys = [k_keys[0]]*len(k_keys)
    if D == 1:
        k_params = vmap(rdm_SE_kernel_params)(jnp.vstack(k_keys))
    elif D == 2:
        k_params = vmap(rdm_SE_kernel_params, in_axes=(0, None, None))(
            jnp.vstack(k_keys), 0.25, 1.)

    # initialize mixing function parameters
    key, mlp_key = jr.split(key, 2)
    mixer_params = init_nica_params(mlp_key, N, M, L, repeat_layers)

    # sample ICs and their mixtures
    key, *sample_keys = jr.split(key, num_samples + 1)
    if tp:
        sample_fun = lambda _: sample_tpnica(_, t, mu_func, kernel_func,
                                             k_params, dfs, mixer_params)
    else:
        sample_fun = lambda _: sample_gpnica(_, t, mu_func, kernel_func, k_params,
                                             mixer_params)
    z, s, tau = lax.map(sample_fun, jnp.vstack(sample_keys))

    # add noise
    x = z + jnp.sqrt(noise_factor)*jr.normal(key, shape=z.shape)
    Q = jnp.eye(M)*noise_factor
    return x, z, s, tau, Q, mixer_params, k_params, dfs
