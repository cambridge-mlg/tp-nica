import jax
import jax.numpy as jnp
from util import mvp, transpose, vdot

def gaussian_standardparams(natparams):
    h, J = natparams
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V,h)
    return mu, V

def gaussian_logZ(natparams):
    h, J = natparams
    J = .5*(J + transpose(J))
    L = jnp.linalg.cholesky(-2*J)
    v = jax.scipy.linalg.solve_triangular(L, h, lower=True)
    halflogdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-1, axis2=-2)), -1)
    return .5*h.shape[-1]*jnp.log(2*jnp.pi) + .5*vdot(v, v) - halflogdet

def gaussian_logZ2(natparams):
    h, J = natparams
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V,h)
    D = h.shape[-1]
    return .5*D*jnp.log(2*jnp.pi) + .5*jnp.linalg.slogdet(V)[1] + .5*vdot(mu, h)