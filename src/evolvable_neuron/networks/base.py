from collections.abc import Iterable
from typing import Callable, Tuple

import numpy as np
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.typing import Array, Initializer
from flax.linen import initializers
from jax import lax

try:
    from evolvable_neuron_plugin import dense
except ModuleNotFoundError:
    dense = linear_relu


def linear_relu(w, b, aux_params, inputs, depth, state):
    s0, s1 = aux_params
    linear_combination = jnp.dot(w, inputs) + b
    return linear_combination if linear_combination > 0 else 0, state


class Dense(nn.Module):
  """A transformation applied over the last dimension of the input.
  """
    out_feats: int
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    depth: int = 0

    @nn.compact
    def __call__(self, inputs: Array) -> Array:

        W = self.param("W", self.kernel_init, (jnp.shape(inputs)[-1], self.out_feats))
        b_vec = self.param("b_vec", self.bias_init, (self.out_feats,))
        state_vec = self.variable(
            "self_updated",
            "state_vec",
            lambda: self.bias_init(self.make_rng("params"), (self.out_feats,)),
        )
        Aux = self.param("Aux", self.bias_init, (2, self.out_feats))

        depth = self.depth

        v_wrapped_dense = jax.vmap(
            lambda w, b, aux_params, inputs, state: dense(
                w, b, aux_params, inputs, depth, state
            ),
            in_axis=1,
            out_axes=1,
        )

        output, state_vec.value = v_wrapped_dense(W, b_vec, Aux, inputs, state_vec)

        return output


class MLP(nn.Module):
    """A multi-layer perceptron module."""

    layer_feats: Iterable[int]
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    depth: int = 0

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for index, out_feats in enumerate(self.layer_feats):
            x = Dense(
                out_feats=out_feats,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                depth=self.depth + index,
                name="dense_%d" % index,
            )(x)
        return x


class Embed(nn.Module):
    """Flax Linen equivalent of Haiku's hk.Embed."""

    #: Number of embeddings in the vocabulary
    vocab_size: int

    #: Dimensionality of each embedding vector
    embed_dim: int

    def __call__(self, inputs):
        embedding = self.param(
            "embedding", nn.initializers.xavier_uniform(), (self.vocab_size, self.embed_dim)
        )

        return embedding[inputs]
