import jax.numpy as jnp


def dense_apply(params, inputs):
    W, b = params
    return jnp.dot(inputs, W) + b