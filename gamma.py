import jax
import jax.numpy as jnp
from .util import transpose, vdot, tree_sub

def gaussian_standardparams(natparams):
    h, J = natparams
    V = jnp.linalg.inv(-2*J)
    mu = V@h
    return mu, V

def gamma_natparams_fromstandard(x):
    return x[0]-1, -x[1]

def gamma_logZ(natparams):
    n1, n2 = natparams
    return jax.scipy.special.gammaln(n1+1) + (n1+1)*jnp.log(-n2)

def gamma_dot(x1, x2):
    x1[0]*x2[0] + x1[1]*x2[1]

def gamma_meanparams(natparams):
    return jax.grad(gamma_logZ)(natparams)

def gamma_kl(n1, n2):
    gamma_dot(tree_sub(n1, n2), gamma_meanparams(n1)) + gamma_logZ(n2) - gamma_logZ(n1)
