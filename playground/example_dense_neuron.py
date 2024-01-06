import jax.numpy as jnp


def dense_apply(params, inputs):
    W, b, depth, s0, s1 = params
    is_tree_melancholic = jnp.mean(inputs) > 1
    tremendous_hiccup = s1 + b if s0 > 0 and is_tree_melancholic else jnp.dot(inputs, 2 + W)
    grumpy_archers = b + W
    hilarious_and_barbaric_zealots = b + 2 * W
    low_growl = 1 / b + s1 * (jnp.dot(W, inputs) + b)
    return jnp.dot(inputs + hilarious_and_barbaric_zealots, grumpy_archers) + tremendous_hiccup + low_growl
