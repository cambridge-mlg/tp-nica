from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as js
import optax

import pdb

from jax import vmap, jit, value_and_grad
from jax.lax import cond, dynamic_slice
from jax.tree_util import tree_map
from util import jax_profiler, rngcall, mvp, tree_get_idx
from utils import rdm_upper_cholesky_of_precision, jax_print, reorder_covmat
from utils import comp_K_N, get_diag_blocks
from tprocess.kernels import compute_K, se_kernel_fn, rdm_SE_kernel_params
from tprocess.kernels import rdm_df
from tprocess.sampling import gen_1d_locations
from nn import init_nica_params, nica_logpx

key = jr.PRNGKey(0)
N = 3
M = 3
n_pseudo = 50
nsamples = 5
T = 200
L = 0
n_data = 1

# prof settings
nsteps = 1e3

# generate locations
t = gen_1d_locations(T)


# initialize generative model params (theta)
theta_tau, key = rngcall(
    lambda _k: vmap(lambda _: rdm_df(_, maxval=20))(jr.split(_k, N)), key
)
theta_k, key = rngcall(
    lambda _k: vmap(lambda _: rdm_SE_kernel_params(_, t)
                   )(jr.split(_k, N)), key
)
theta_var, key = rngcall(lambda _: jr.uniform(_, shape=(M,),
            minval=-1, maxval=1), key)
theta_mix, key = rngcall(lambda _: init_nica_params(
    _, N, M, L, repeat_layers=False), key)
theta_x = (theta_mix, theta_var)

theta = (theta_x, theta_k, theta_tau)


# initialize variational parameters (phi) with pseudo-points (tu)
tu, key = rngcall(lambda _: vmap(lambda k: jr.choice(k, t,
        shape=(n_pseudo,), replace=False))(jr.split(_, n_data)), key)
tu = tu[0]

W, key = rngcall(lambda _k: vmap(
    lambda _: rdm_upper_cholesky_of_precision(_, N)*10, out_axes=-1)(
        jr.split(_k, n_pseudo)), key
)
phi_s = (jnp.repeat(W[None, :], n_data, 0).squeeze(),
         jnp.ones(shape=(n_data, N, n_pseudo)).squeeze(), tu)
phi_df, key = rngcall(lambda _: vmap(rdm_df)(jr.split(_, n_data*N)), key)
phi_df = phi_df.reshape(n_data, N)
phi_tau = (phi_df, phi_df*10)
phi = (phi_s, phi_tau)

# inference set up
theta_x, theta_cov = theta[:2]
What, yhat, tu = phi_s
N, n_pseudo = yhat.shape
T = t.shape[0]


# profiling
# 1
theta = (se_kernel_fn, theta_cov, N, tu)


def Kuu_prof(key, theta, s):
    _ = key
    cov_fn, theta_cov, N, tu = theta
    Kuu = jax.block_until_ready(vmap(lambda b: vmap(lambda a:
         comp_K_N(a, b, cov_fn, theta_cov)+1e-6*jnp.eye(N)
     )(tu))(tu))
    s = s + 1e-16
    return s, Kuu+s


Kuu = jax.block_until_ready(vmap(lambda b: vmap(lambda a:
         comp_K_N(a, b, se_kernel_fn, theta_cov)+1e-6*jnp.eye(N)
     )(tu))(tu))
Kuu_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), Kuu_prof, nsteps, 3))

# 2
theta = (Kuu, N, n_pseudo)


def Kuuswap_prof(key, theta, s):
    _ = key
    Kuu, N, n_pseudo= theta
    Kuu = jax.block_until_ready(Kuu.swapaxes(1, 2).reshape(n_pseudo*N,
                                                           n_pseudo*N))
    s = s + 1e-16
    return s, Kuu+s


Kuu = jax.block_until_ready(Kuu.swapaxes(1, 2).reshape(n_pseudo*N, n_pseudo*N))
Kuuswap_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), Kuuswap_prof, nsteps, 3))


# 3
theta = (se_kernel_fn, theta_cov, tu, t)


def Ksu_prof(key, theta, s):
    _ = key
    cov_fn, theta_cov, tu, t = theta
    Ksu = jax.block_until_ready(vmap(lambda b:vmap(lambda a:
        comp_K_N(a, b, cov_fn, theta_cov))(tu))(t))
    s = s + 1e-16
    return s, Ksu+s


Ksu = jax.block_until_ready(vmap(lambda b:vmap(lambda a:
    comp_K_N(a, b, se_kernel_fn, theta_cov))(tu))(t))
Ksu_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), Ksu_prof, nsteps, 3))


# 4 
theta = (Ksu, T, N, n_pseudo)


def Ksuswap_prof(key, theta, s):
    _ = key
    Ksu, T, N, n_pseudo = theta
    Ksu = jax.block_until_ready(Ksu.swapaxes(1, 2).reshape(T*N, n_pseudo*N))
    s = s + 1e-16
    return s, Ksu+s

Ksu = jax.block_until_ready(Ksu.swapaxes(1, 2).reshape(T*N, n_pseudo*N))
Ksuswap_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), Ksuswap_prof, nsteps, 3))


# 5
theta = (se_kernel_fn, theta_cov, t)


def Kss_prof(key, theta, s):
    _ = key
    cov_fn, theta_cov, t = theta 
    Kss = jax.block_until_ready(vmap(lambda tc: vmap(
        lambda t: cov_fn(t, t, tc))(t))(theta_cov))
    s = s + 1e-16
    return s, Kss+s


kss = jax.block_until_ready(vmap(lambda tc: vmap(
        lambda t: se_kernel_fn(t, t, tc))(t))(theta_cov))
Kss_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), Kss_prof, nsteps, 3))


# 6
theta = (What, yhat)


def WTy_prof(key, theta, s):
    _ = key
    What, yhat = theta 
    WTy = jax.block_until_ready(jnp.einsum('ijk,ik->jk',
                                           What, yhat).T.reshape(-1, 1))
    s = s + 1e-16
    return s, WTy+s


WTy = jax.block_until_ready(jnp.einsum('ijk,ik->jk',
                                       What, yhat).T.reshape(-1, 1))
WTy_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), WTy_prof, nsteps, 3))



# 7
theta = What


def L_prof(key, theta, s):
    _ = key
    What = theta 
    L = jax.block_until_ready(js.linalg.block_diag(*jnp.moveaxis(
      jnp.einsum('ijk, ilk->jlk', What, What), -1, 0)))
    s = s + 1e-16
    return s, L+s


L = jax.block_until_ready(js.linalg.block_diag(*jnp.moveaxis(
  jnp.einsum('ijk, ilk->jlk', What, What), -1, 0)))
L_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), L_prof, nsteps, 3))


# 8
theta = (L, Kuu)


def LK_prof(key, theta, s):
    _ = key
    L, Kuu = theta 
    L = jax.block_until_ready(L@Kuu)
    s = s + 1e-16
    return s, L+s


LK = L@Kuu
LK_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), LK_prof, nsteps, 3))


# 9
theta = (L, LK)


def lufact_prof(key, theta, s):
    _ = key
    L, LK = theta
    lu_fact = jax.block_until_ready(js.linalg.lu_factor(jnp.eye(L.shape[0])+LK))
    s = s + 1e-16
    return s, lu_fact[0]+s


lu_fact = jax.block_until_ready(js.linalg.lu_factor(jnp.eye(L.shape[0])+LK))
lufact_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), lufact_prof, nsteps, 3))


# 10
theta = (lu_fact, WTy)


def KyyWTy_prof(key, theta, s):
    _ = key
    lu_fact, WTy = theta
    KyyWTy = jax.block_until_ready(js.linalg.lu_solve(lu_fact, WTy))
    s = s + 1e-16
    return s, KyyWTy+s


KyyWTy = jax.block_until_ready(js.linalg.lu_solve(lu_fact, WTy))
KyyWTy_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), KyyWTy_prof, nsteps, 3))


# 11
theta = (Ksu, KyyWTy)


def mu_s_prof(key, theta, s):
    _ = key
    Ksu, KyyWTy = theta
    mu_s = jax.block_until_ready(Ksu@KyyWTy)
    s = s + 1e-16
    return s, mu_s+s


mu_s = jax.block_until_ready(Ksu@KyyWTy)
mu_s_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), mu_s_prof, nsteps, 3))


# 12
theta = (lu_fact, L, Ksu, kss, T, N)


def cov_s_prof(key, theta, s):
    _ = key
    lu_fact, L, Ksu, kss, T, N = theta
    cov_s = jax.block_until_ready(vmap(
        lambda X, y: jnp.diag(y)-X@js.linalg.lu_solve(lu_fact, L)@X.T,
          in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss))
    s = s + 1e-16
    return s, cov_s+s


cov_s = jax.block_until_ready(vmap(
        lambda X, y: jnp.diag(y)-X@js.linalg.lu_solve(lu_fact, L)@X.T,
          in_axes=(0, -1))(Ksu.reshape(T, N, -1), kss))
cov_s_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), cov_s_prof, nsteps, 3))


# 13
theta = (mu_s, T, N, nsamples)


def s_prof(key, theta, s):
    mu_s, T, N, nsamples = theta
    sest, _ = jax.block_until_ready(rngcall(
        lambda _:jr.multivariate_normal(_, mu_s.reshape(T, N),
        cov_s, shape=(nsamples, T)), key))
    s = s + 1e-16
    return s, sest+s


s, key = jax.block_until_ready(rngcall(
    lambda _:jr.multivariate_normal(_, mu_s.reshape(T, N),
    cov_s, shape=(nsamples, T)), key))
x, key = jax.block_until_ready(rngcall(
    lambda _:jr.multivariate_normal(_, mu_s.reshape(T, N),
    cov_s, shape=(nsamples, T)), key))
x = x[0].T
s_time = jax.block_until_ready(jax_profiler(
   theta, jnp.array(1e-16), s_prof, nsteps, 3))


# 14
theta = (nica_logpx, theta_x, x, s)


def Elogpx_prof(key, theta, s):
    logpx, theta_x, x, sest = theta
    Elogpx = jax.block_until_ready(jnp.mean(
        jnp.sum(vmap(lambda _: vmap(logpx, (1, 0, None))(x, _, theta_x))(sest), 1)
    ))
    s = s + 1e-16
    return s, Elogpx+s

Elogpx = jax.block_until_ready(jnp.mean(
    jnp.sum(vmap(lambda _: vmap(nica_logpx, (1, 0, None))(x, _, theta_x))(s), 1)))
Elogpx_time = jax.block_until_ready(jax_profiler(
   theta, jnp.array(1e-16), Elogpx_prof, nsteps, 3))


# 15
theta = (lu_fact, LK)


def tr_prof(key, theta, s):
    lu_fact, LK = theta
    tr = jax.block_until_ready(jnp.trace(js.linalg.lu_solve(lu_fact, LK.T, trans=1).T))
    s = s + 1e-16
    return s, tr+s


tr = jax.block_until_ready(jnp.trace(js.linalg.lu_solve(lu_fact, LK.T, trans=1).T))
tr_time = jax.block_until_ready(jax_profiler(
   theta, jnp.array(1e-16), tr_prof, nsteps, 3))


# 16
theta = (Kuu, KyyWTy)


def h_prof(key, theta, s):
    _ = key
    Kuu, KyyWTy = theta
    h = jax.block_until_ready(Kuu@KyyWTy)
    s = s + 1e-16
    return s, h+s

h = jax.block_until_ready(Kuu@KyyWTy)
h_time = jax.block_until_ready(jax_profiler(
   theta, jnp.array(1e-16), h_prof, nsteps, 3))


# 17
theta = (WTy, h, L, LK)


def logz_prof(key, theta, s):
    _ = key
    WTy, h, L, LK = theta
    logZ = jax.block_until_ready(-0.5*(-jnp.dot(WTy.squeeze(), h)
                 +jnp.linalg.slogdet(jnp.eye(L.shape[0])+LK)[1]))
    s = s + 1e-16
    return s, logZ+s


logZ = jax.block_until_ready(-0.5*(-jnp.dot(WTy.squeeze(), h)
             +jnp.linalg.slogdet(jnp.eye(L.shape[0])+LK)[1]))
logz_time = jax.block_until_ready(jax_profiler(
   theta, jnp.array(1e-16), logz_prof, nsteps, 3))


# 18 
theta = (tr, h, L, WTy, logZ)


def KL_prof(key, theta, s):
    _ = key
    tr, h, L, WTy, logZ = theta
    KL = jax.block_until_ready(-0.5*(tr+h.T@L@h)+WTy.T@h - logZ)
    s = s + 1e-16
    return s, KL+s


KL = jax.block_until_ready(-0.5*(tr+h.T@L@h)+WTy.T@h - logZ)
KL_time = jax.block_until_ready(jax_profiler(
   theta, jnp.array(1e-16), KL_prof, nsteps, 3))


# Xtra
theta = (lu_fact, L, Ksu, kss, T, N)

def cov_s2_prof(key, theta, s):
    _ = key
    lu_fact, L, Ksu, kss, T, N = theta
    cov_s = jax.block_until_ready(vmap(jnp.diag, 1)(kss) - get_diag_blocks(
        Ksu@js.linalg.lu_solve(lu_fact, L@Ksu.T), N, T))
    s = s + 1e-16
    return s, cov_s+s


cov_s2_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), cov_s2_prof, nsteps, 3))


def cov_s3_prof(key, theta, s):
    _ = key
    lu_fact, L, Ksu, kss, T, N = theta
    solved = js.linalg.lu_solve(lu_fact, L@Ksu.T)
    cov_s = jax.block_until_ready(vmap(
        lambda c, A, B: jnp.diag(c)-A@B,
          in_axes=(-1, 0, 1))(kss, Ksu.reshape(T, N, -1),
                              solved.reshape(-1, T, N)))
    s = s + 1e-16
    return s, cov_s+s


cov_s3_time = jax.block_until_ready(jax_profiler(
    theta, jnp.array(1e-16), cov_s3_prof, nsteps, 3))






# print
print("Kuu: ", Kuu_time)
print("Kuuswap: ", Kuuswap_time)
print("Ksu :", Ksu_time)
print("Ksuswap_time: ", Ksuswap_time)
print("Kss_time: ", Kss_time)
print("WTy_time: ", WTy_time)
print("L_time: ", L_time)
print("LK_time: ", LK_time)
print("lufact_time: ", lufact_time)
print("KyyWTy_time: ", KyyWTy_time)
print("mu_s_time: ", mu_s_time)
print("cov_s_time: ", cov_s_time)
print("cov_s2_time: ", cov_s2_time)
print("cov_s3_time: ", cov_s3_time)
print("s_time: ", s_time)
print("Elogpx_time: ", Elogpx_time)
print("tr_time: ", tr_time)
print("h_time: ", h_time)
print("logz_time: ", logz_time)
print("KL_time: ", KL_time)
