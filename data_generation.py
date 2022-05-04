import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd

import pdb

from jax import jit, vmap
from jax.tree_util import Partial, tree_map
from nn import init_nica_params, nica_mlp
from utils import gaussian_sample_from_mu_prec, invmp

import matplotlib.pyplot as plt
import seaborn as sns


def euclid_dist(x, y):
    return jnp.linalg.norm(x-y)


def SE_kernel(params):
    def se_kernel_fn(x, y):
        scale, loc = params
        return scale**2 * jnp.exp(-0.5*euclid_dist(x, y)**2 / loc**2)
    return se_kernel_fn


def zero_mean_fn(x):
    return jnp.zeros(x.shape[0])


def cos_1d_mean_fn(x):
    return jnp.cos(x)


def compute_K(x, kernel_fn):
    return vmap(lambda _x: vmap(lambda _y: kernel_fn(_x, _y))(x))(x)


def euclid_dist_mat(x):
    return vmap(lambda _x: vmap(lambda _y: euclid_dist(_x, _y))(x))(x)


def gen_rdm_SE_kernel_params(key, x, lscale_min_multip=0.01,
                             lscale_max_multip=2.5, var_min=0.1, var_max=1.):
    """
    Note: x is needed to find reasonable length-scale thats not too smooth
    """
    D = euclid_dist_mat(x)
    min_val = jnp.min(D[jnp.tril_indices(D.shape[0], k=-1)])
    lscale = jr.uniform(key, minval=lscale_min_multip*min_val,
                        maxval=lscale_max_multip*min_val)
    var_key, _ = jr.split(key)
    var = jr.uniform(var_key, minval=var_min, maxval=var_max)
    return (var, lscale)


def gen_rdm_Gamma_params(key, max_val=5):
    """
    Note: df > 2 has been hard-coded
    Note: max_val is inclusive
    """
    key = jr.split(key)
    params = jr.choice(key, a=jnp.arange(start=3, stop=max_val+1), shape=(2,))
    return (params[0], params[1])


def sample_tprocess(key, latent_inputs, gp_mu_fn, gp_kernel_fn, gamma_params):
    # sample (1/r) from Gamma prior
    nu, rho = gamma_params
    r_key, t_key = jr.split(key)
    r_inv_sample = tfd.Gamma(nu/2, rho/2).sample(1, seed=r_key)
    # define GP parameters
    mu = gp_mu_fn(latent_inputs)
    K = compute_K(latent_inputs, gp_kernel_fn)
    # sample
    t_sample = tfd.MultivariateNormalFullCovariance(mu,
        (1/r_inv_sample)*K).sample(1, seed=t_key)
    return t_sample


def multi_tprocess_sampler(key, x, gp_mu_fn_list, gp_k_fn_list,
                           gamma_params_list):
    # sample t-process for each component
    N = len(gp_mu_fn_list)
    sample_keys = [jr.fold_in(key, _) for _ in range(N)]
    s_list = tree_map(lambda a, b, c, d: sample_tprocess(a, x, b, c, d),
        sample_keys, gp_mu_fn_list, gp_k_fn_list, gamma_params_list)
    s = jnp.vstack(s_list)
    return s


def sample_tpnica(key, x, gp_mu_fn_list, gp_k_fn_list, gamma_params_list,
                  mixer_params):
    # sample each IC as a t-process
    key, s_key = jr.split(key)
    s = multi_tprocess_sampler(s_key, x, gp_mu_fn_list, gp_k_fn_list,
                               gamma_params_list)
    # mix the ICs
    z = vmap(nica_mlp, (None, 1), 1)(mixer_params, s)
    # add output noise (scaled to be appropriate size)
    M = z.shape[0]
    z_vars = jnp.diag(jnp.cov(z))
    key, noise_key = jr.split(key)
    noise_vars = jr.uniform(noise_key, shape=(M,), minval=0.05*z_vars,
                            maxval=0.2*z_vars)
    # get precision matrix from cov matrix
    R = invmp(noise_vars*jnp.eye(M), jnp.eye(M))
    y_keys = jr.split(key, z.shape[1])
    y = vmap(gaussian_sample_from_mu_prec, (1, None, 0), 1)(z, R, y_keys)
    return y, z, s, R


def gen_tprocess_nica_data(x, N, M, L, num_samples, repeat_layers=False):
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

        # sample ICs
        key, *sample_keys = jr.split(key, num_samples+1)
        y, z, s, R = vmap(
            lambda _: sample_tpnica(_, x, mu_funcs, k_funcs,
                                    gamma_params, mixer_params)
        )(jnp.vstack(sample_keys))
        pdb.set_trace()
        return 0
    return _sample


def gen_1d_locations(t):
    return jnp.linspace(0, 10, t)[:, None]


def gen_2d_locations(t):
    X, Y = jnp.meshgrid(jnp.linspace(0, 10, int(jnp.sqrt(t))),
                        jnp.linspace(0, 10, int(jnp.sqrt(t))))
    return jnp.hstack((X.flatten()[:, None], Y.flatten()[:, None]))


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
















