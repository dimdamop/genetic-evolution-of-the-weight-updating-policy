# Copyright 2024 Dimitrios Damopoulos. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
"""Some basic :class:`flax.linen.Module` layers. Also, very importantly, this is where the rest of
:mod:`evolvable_neuron` should import the evolvable definition of a dense layer from.
"""

from collections.abc import Iterable
from typing import Callable, Tuple

import flax.linen as nn
import jax
import numpy as np
from flax.linen import initializers
from flax.typing import Array, Initializer
from jax import lax
from jax import numpy as jnp


def linear_relu(w, b, aux_params, inputs, depth, state):
    s0, s1 = aux_params
    linear_combination = jnp.dot(w, inputs) + b
    return jnp.where(linear_combination > 0, linear_combination, 0), state


try:
    from evolvable_neuron_plugin import dense
except ModuleNotFoundError:
    dense = linear_relu


class Dense(nn.Module):
    """A transformation applied over the last dimension of the input."""

    out_feats: int
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    depth: int = 0

    @nn.compact
    def __call__(self, inputs: Array) -> Array:

        W = self.param("W", self.kernel_init, (self.out_feats, jnp.shape(inputs)[-1]))
        b_vec = self.param("b_vec", self.bias_init, (self.out_feats,))

        # `state_vec` is not supposed to be updated via gradient descent. Rather, it is updated
        # iteratively by :func:`dense` itself by some fixed transformation which is subject to
        # evolution (like everything in :func:`dense`).
        state_vec = self.variable(
            "self_updated",
            "state_vec",
            lambda shape: self.bias_init(self.make_rng("params"), shape),
            (self.out_feats,)
        )
        Aux = self.param("Aux", self.bias_init, (self.out_feats, 2))
        depth = self.depth

        multi_output_dense = jax.vmap(
            lambda w, b, aux_params, state: dense(w, b, aux_params, inputs, depth, state),
        )

        output, state_vec.value = multi_output_dense(W, b_vec, Aux, state_vec.value)

        return output.T


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

    @nn.compact
    def __call__(self, inputs):
        embedding = self.param(
            "embedding", nn.initializers.xavier_uniform(), (self.vocab_size, self.embed_dim)
        )

        return embedding[inputs]
