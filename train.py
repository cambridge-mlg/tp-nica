from jax.config import config

config.update("jax_enable_x64", True)

#import jax
#import jax.numpy as jnp
#import jax.random as jr
#import optax
#
#from jax import vmap, jit, value_and_grad, lax
#from jax.lax import cond
#from optax import chain, piecewise_constant_schedule, scale_by_schedule
#
#
#def train(y, z, s, states, params, args, est_key):
#    N = args.n
#    M = args.m
#    d = args.d
#    T = args.t
#    return 0
