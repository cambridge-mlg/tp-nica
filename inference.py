import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import jax.debug as jdb
import pdb

from jax import vmap, jit, lax
from jax.tree_util import tree_map, Partial

from kernels import (
    bound_se_kernel_params,
    squared_euclid_dist_mat
)
from utils import (
    custom_lu_solve,
    custom_trans_lu_solve,
    jax_print,
    fill_tril,
    K_N_diag,
    K_TN_blocks,
    quad_form
)
from util import *
from gamma import *


def approx_cov(kss_diag_t, Ksu_t, Kyy_L):
    return kss_diag_t-Ksu_t@Kyy_L@Ksu_t.T


def structured_elbo_s(key, theta, phi_s, logpx, cov_fn, x, t, tau, nsamples):
    theta_x, theta_cov = theta[:2]
    W, m, tu = phi_s
    N, n_pseudo = m.shape
    T = t.shape[0]
    theta_cov = tree_map(jnp.exp, theta_cov)
    # repeat in case the same kernel replicated for all ICs
    theta_cov = tree_map(lambda _: _.repeat(N-theta_cov[0].shape[0]+1),
                         theta_cov)
    t_dist_mat = jnp.sqrt(squared_euclid_dist_mat(t))
    theta_cov = bound_se_kernel_params(
        theta_cov, sigma_min=1e-3,
        ls_min=jnp.min(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)]),
        ls_max=jnp.max(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)])
    )

    Kuu = K_TN_blocks(tu, tu, cov_fn, theta_cov, tau)
    Kuu = Kuu.swapaxes(1, 2).reshape(n_pseudo, N, -1)
    Ksu = K_TN_blocks(t, tu, cov_fn, theta_cov, tau)
    Ksu = Ksu.swapaxes(1, 2).reshape(T, N, -1)
    kss = vmap(K_N_diag, in_axes=(0, 0, None, None, None))(
       t, t, cov_fn, theta_cov, tau)

    # compute parameters for \tilde{q(s|tau)}
    W = vmap(fill_tril, in_axes=(1, None))(W, N)
    L = jnp.matmul(W, W.swapaxes(1, 2))
    LK = jnp.matmul(L, Kuu).reshape(-1, Kuu.shape[-1])
    Jyy = jnp.eye(LK.shape[0])+LK
    lu_fact = js.linalg.lu_factor(Jyy)
    Kyy_m = custom_lu_solve(Jyy, m.T.reshape(-1), lu_fact)
    mu_s = jnp.matmul(Ksu, Kyy_m)
    Kyy_L = custom_lu_solve(Jyy, js.linalg.block_diag(*L), lu_fact)
    cov_s = vmap(approx_cov, in_axes=(0, 0, None))(kss, Ksu, Kyy_L)
    s = jr.multivariate_normal(key, mu_s, cov_s, shape=(nsamples, T))

    # compute E_{\tilde{q(s|tau)}}[log_p(x_t|s_t)]
    Elogpx = jnp.sum(vmap(vmap(logpx, (1, 0, None)), (None, 0, None))(
            x, s, theta_x), 1).mean()

    # compute KL[q(u)|p(u)]
    h = Kuu.reshape(-1, Kuu.shape[-1])@Kyy_m
    logZ = 0.5*(jnp.dot(m.T.reshape(-1), h) - jnp.linalg.slogdet(Jyy)[1])
    tr = jnp.trace(Kuu.reshape(-1, Kuu.shape[-1])@Kyy_L)
    h = h.reshape(n_pseudo, -1)
    KLqpu = -0.5*(tr + vmap(quad_form)(h, L).sum()) + jnp.dot(
        m.T.reshape(-1), h.reshape(-1)) - logZ
    return Elogpx-KLqpu, s


# compute elbo estimate, assumes q(tau) is gamma
def structured_elbo(key, theta, phi, logpx, cov_fn, x, t, nsamples):
    nsamples_s, nsamples_tau = nsamples
    theta_tau = theta[2]
    theta_tau = 2.+jnp.exp(theta_tau)
    phi_s, phi_tau = phi[:2]
    N = phi_tau[0].shape[0]
    # in case df param is replicated to be same for all ICs
    theta_tau = theta_tau.repeat(N-theta_tau.shape[0]+1)
    # to avoid numerical issues
    phi_tau = tree_map(jnp.exp, phi_tau)
    tau, key = rngcall(gamma_sample, key, gamma_natparams_fromstandard(phi_tau),
                       (nsamples_tau, *phi_tau[0].shape))
    kl = jnp.sum(
        gamma_kl(
            gamma_natparams_fromstandard(phi_tau),
            gamma_natparams_fromstandard((theta_tau/2, theta_tau/2))), 0)
    vlb_s, s = vmap(lambda _: structured_elbo_s(
        key, theta, phi_s, logpx, cov_fn, x, t, _, nsamples_s))(tau)
    return jnp.mean(vlb_s, 0) - kl, s


#@jit
def gp_elbo(key, theta, phi_s, logpx, cov_fn, x, t, nsamples):
    theta_x, theta_cov = theta[:2]
    What, yhat, tu = phi_s
    N, n_pseudo = yhat.shape
    T = t.shape[0]
    theta_cov = tree_map(jnp.exp, theta_cov)
    # repeat in case the same kernel replicated for all ICs
    theta_cov = tree_map(lambda _: _.repeat(N-theta_cov[0].shape[0]+1),
                         theta_cov)
    t_dist_mat = jnp.sqrt(squared_euclid_dist_mat(t))
    theta_cov = bound_se_kernel_params(
        theta_cov, sigma_min=1e-3,
        ls_min=jnp.min(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)]),
        ls_max=jnp.max(t_dist_mat[jnp.triu_indices_from(t_dist_mat, k=1)])
    )
    Kuu = vmap(lambda b: vmap(lambda a:
        comp_K_N(a, b, cov_fn, theta_cov)
    )(tu))(tu)
    Kuu = Kuu.swapaxes(1, 2).reshape(n_pseudo*N, n_pseudo*N)
    Ksu = vmap(lambda b:vmap(lambda a:
        comp_K_N(a, b, cov_fn, theta_cov))(tu))(t)
    Ksu = Ksu.swapaxes(1, 2).reshape(T*N, n_pseudo*N)
    kss = vmap(
        lambda tc: vmap(lambda t: cov_fn(t, t, tc))(t)
        )(theta_cov)

    # compute parameters for \tilde{q(s)}
    What = vmap(fill_triu, in_axes=(1, None), out_axes=-1)(What, N)
    WTy = jnp.einsum('ijk,ik->jk', What, yhat).T.reshape(-1, 1)
    L = js.linalg.block_diag(*jnp.moveaxis(
      jnp.einsum('ijk, ilk->jlk', What, What), -1, 0))
    LK = L@Kuu
    lu_fact = jit(js.linalg.lu_factor)(jnp.eye(L.shape[0])+LK)
    KyyWTy = custom_lu_solve(jnp.eye(L.shape[0])+LK, WTy, lu_fact)
    #KyyWTy = js.linalg.lu_solve(lu_fact, WTy)
    mu_s = Ksu @ KyyWTy
    cov_solve = custom_lu_solve(jnp.eye(L.shape[0])+LK, L, lu_fact)
    cov_s = vmap(lambda X, y: jnp.diag(y)-X@cov_solve@X.T,
          in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss)
    #cov_s = vmap(lambda X, y: jnp.diag(y)-X@js.linalg.lu_solve(lu_fact, L)@X.T,
    #      in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss)
    s, key = rngcall(lambda _: jr.multivariate_normal(_, mu_s.reshape(T, N),
        cov_s, shape=(nsamples, T)), key)

    # compute E_{\tilde{q(s)}}[log_p(x_t|s_t)]
    Elogpx = jnp.mean(
        jnp.sum(vmap(lambda _: vmap(logpx,(1, 0, None))(x, _, theta_x))(s), 1)
    )

    # compute KL[q(u)|p(u)]
    tr = jnp.trace(js.linalg.lu_solve(lu_fact, LK.T, trans=1).T)
    h = Kuu@KyyWTy
    logZ = 0.5*(jnp.dot(WTy.squeeze(), h)
                -jnp.linalg.slogdet(jnp.eye(L.shape[0])+LK)[1])
    KLqpu = -0.5*(tr+h.T@L@h)+WTy.T@h - logZ
    return Elogpx-KLqpu, s


def avg_neg_elbo(key, theta, phi_n, logpx, cov_fn, x, t, nsamples, elbo_fn):
    """
    Calculate average negative elbo over training samples
    """
    vlb, s = vmap(elbo_fn, (0, None, 0, None, None, 0, None, None))(
        jr.split(key, x.shape[0]), theta, phi_n, logpx, cov_fn, x, t, nsamples)
    return -vlb.mean(), s





avg_neg_tp_elbo = Partial(avg_neg_elbo, elbo_fn=structured_elbo)
avg_neg_gp_elbo = Partial(avg_neg_elbo, elbo_fn=gp_elbo)


if __name__ == "__main__":
    pdb.set_trace()
