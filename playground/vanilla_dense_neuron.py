import jax.numpy as jnp


def dense_apply(params, inputs):
    W, b, depth, s0, s1 = params
    return jnp.dot(inputs, W) + b
