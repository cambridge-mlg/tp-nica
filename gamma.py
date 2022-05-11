import jax
import jax.numpy as jnp
from util import transpose, vdot, tree_sub

def gamma_logZ(natparams):
    n1, n2 = natparams
    return jax.scipy.special.gammaln(n1+1) - (n1+1)*jnp.log(-n2)

def gamma_dot(x1, x2):
    return x1[0]*x2[0] + x1[1]*x2[1]

def gamma_stats(x):
    return jnp.log(x), x

def gamma_meanparams(natparams):
    n1, n2 = natparams
    return jax.scipy.special.digamma(n1+1) - jnp.log(-n2), -(n1+1)/n2

def gamma_kl(n1, n2):
    return gamma_dot(tree_sub(n1, n2), gamma_meanparams(n1)) + gamma_logZ(n2) - gamma_logZ(n1)

def gamma_logprob(natparams, x):
    n1, n2 = natparams
    a, b = n1+1, -n2
    return a*jnp.log(b) - jax.scipy.special.gammaln(a) + (a-1)*jnp.log(x) - b*x

def gamma_mean(natparams):
    n1, n2 = natparams
    return -(n1+1)/n2

def gamma_var(natparams):
    n1, n2 = natparams
    return (n1+1)/(n2**2)