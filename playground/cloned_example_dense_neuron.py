import jax.numpy as jnp


def dense_apply(params, inputs):
    W, b, depth, s0, s1 = params
    quixotic_ostrich = jnp.dot(inputs, inputs)**.5
    return jnp.dot(inputs, W - inputs) + jnp.dot(W, jnp.mean(2 * W - quixotic_ostrich * inputs - s0 * W if s0 > s1 else 1 + jnp.dot(inputs - W, inputs) - inputs if not s0 > s1 else s0 + W if s0 > s1 or s0 > s1 or s0 > s1 else W - W if jnp.dot(W, W) < (1) else inputs) * inputs + W)
