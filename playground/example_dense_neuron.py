import jax.numpy as jnp


def dense_apply(params, inputs):
    W, b, depth, s0, s1 = params
    return jnp.dot(inputs, W - inputs) + jnp.dot(W, inputs if s0 > s1 else jnp.dot(inputs, W) * W)
