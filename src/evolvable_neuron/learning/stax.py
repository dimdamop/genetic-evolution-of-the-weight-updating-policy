# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""A module that constructs standard layers of deep networks.

Following the convention used in `Keras` and `tf.layers`, we use `CamelCase` for the names of
layer constructors, like `Conv` and `Relu`. For other functions, we follow the standard `snake_case`
Python practice (eg.: `lax.conv` and `relu`).

Each layer constructor function returns a pair of functions `init` and `apply`, where:

 * `init`: takes an rng key and an input shape and returns an `(output_shape, params)` pair;

 * `apply`: takes params, inputs, and an rng key and applies the layer.

For clarity, these two functions are returned as elements of a named tuple.
"""

import functools
import logging
import operator as op
from collections import namedtuple
from typing import Sequence

import jax.numpy as jnp
from jax import lax, random
from jax.nn import (
    elu,
    gelu,
    leaky_relu,
    log_softmax,
    relu,
    selu,
    sigmoid,
    softmax,
    softplus,
    standardize,
)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

LayerT = namedtuple("LayerT", ["init", "apply"])


def Dense(out_dim, W_init=glorot_normal(), b_init=normal()) -> LayerT:
    """Layer constructor function for a dense (aka. fully-connected) layer."""

    def init(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply(params, inputs, **kwargs):
        W, b = params
        return jnp.dot(inputs, W) + b

    return LayerT(init=init, apply=apply)


def GeneralConv(
    dimension_numbers,
    out_chan,
    filter_shape,
    strides=None,
    padding="VALID",
    W_init=None,
    b_init=normal(1e-6),
):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot_normal(rhs_spec.index("I"), rhs_spec.index("O"))

    def init(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [
            out_chan
            if c == "O"
            else input_shape[lhs_spec.index("C")]
            if c == "I"
            else next(filter_shape_iter)
            for c in rhs_spec
        ]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers
        )
        bias_shape = [out_chan if c == "C" else 1 for c in out_spec]
        k1, k2 = random.split(rng)
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
        return output_shape, (W, b)

    def apply(params, inputs, **kwargs):
        W, b = params
        return (
            lax.conv_general_dilated(
                inputs, W, strides, padding, one, one, dimension_numbers=dimension_numbers
            )
            + b
        )

    return LayerT(init=init, apply=apply)


Conv = functools.partial(GeneralConv, ("NHWC", "HWIO", "NHWC"))


def GeneralConvTranspose(
    dimension_numbers,
    out_chan,
    filter_shape,
    strides=None,
    padding="VALID",
    W_init=None,
    b_init=normal(1e-6),
):
    """Layer construction function for a general transposed-convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot_normal(rhs_spec.index("I"), rhs_spec.index("O"))

    def init(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [
            out_chan
            if c == "O"
            else input_shape[lhs_spec.index("C")]
            if c == "I"
            else next(filter_shape_iter)
            for c in rhs_spec
        ]
        output_shape = lax.conv_transpose_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers
        )
        bias_shape = [out_chan if c == "C" else 1 for c in out_spec]
        k1, k2 = random.split(rng)
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
        return output_shape, (W, b)

    def apply(params, inputs, **kwargs):
        W, b = params
        return (
            lax.conv_transpose(inputs, W, strides, padding, dimension_numbers=dimension_numbers) + b
        )

    return LayerT(init=init, apply=apply)


Conv1DTranspose = functools.partial(GeneralConvTranspose, ("NHC", "HIO", "NHC"))
ConvTranspose = functools.partial(GeneralConvTranspose, ("NHWC", "HWIO", "NHWC"))


def BatchNorm(
    axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True, beta_init=zeros, gamma_init=ones
):
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()

    if jnp.isscalar(axis):
        axis = (axis,)

    def init(rng, input_shape):
        shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
        k1, k2 = random.split(rng)
        beta, gamma = _beta_init(k1, shape), _gamma_init(k2, shape)
        return input_shape, (beta, gamma)

    def apply(params, x, **kwargs):
        """
        Args:
            params: beta, gamma
        """
        beta, gamma = params
        # TODO(phawkins): jnp.expand_dims should accept an axis tuple.
        # (https://github.com/numpy/numpy/issues/12290)
        ed = tuple(None if i in axis else slice(None) for i in range(jnp.ndim(x)))
        z = standardize(x, axis, epsilon=epsilon)

        if center and scale:
            return gamma[ed] * z + beta[ed]
        if center:
            return z + beta[ed]
        if scale:
            return gamma[ed] * z
        return z

    return LayerT(init=init, apply=apply)


def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init = lambda rng, input_shape: (input_shape, ())
    apply = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return LayerT(init=init, apply=apply)


Tanh = elementwise(jnp.tanh)
Relu = elementwise(relu)
Exp = elementwise(jnp.exp)
LogSoftmax = elementwise(log_softmax, axis=-1)
Softmax = elementwise(softmax, axis=-1)
Softplus = elementwise(softplus)
Sigmoid = elementwise(sigmoid)
Elu = elementwise(elu)
LeakyRelu = elementwise(leaky_relu)
Selu = elementwise(selu)
Gelu = elementwise(gelu)


def _pooling_layer(reducer, init_val, rescaler=None):
    def PoolingLayer(window_shape, strides=None, padding="VALID", spec=None):
        """Layer construction function for a pooling layer."""
        strides = strides or (1,) * len(window_shape)
        rescale = rescaler(window_shape, strides, padding) if rescaler else None

        if spec is None:
            non_spatial_axes = 0, len(window_shape) + 1
        else:
            non_spatial_axes = spec.index("N"), spec.index("C")

        for i in sorted(non_spatial_axes):
            window_shape = window_shape[:i] + (1,) + window_shape[i:]
            strides = strides[:i] + (1,) + strides[i:]

        def init(rng, input_shape):
            padding_vals = lax.padtype_to_pads(input_shape, window_shape, strides, padding)
            ones = (1,) * len(window_shape)
            out_shape = lax.reduce_window_shape_tuple(
                input_shape, window_shape, strides, padding_vals, ones, ones
            )
            return out_shape, ()

        def apply(params, inputs, **kwargs):
            out = lax.reduce_window(inputs, init_val, reducer, window_shape, strides, padding)
            return rescale(out, inputs, spec) if rescale else out

        return LayerT(init=init, apply=apply)

    return PoolingLayer


MaxPool = _pooling_layer(lax.max, -jnp.inf)
SumPool = _pooling_layer(lax.add, 0.0)


def _normalize_by_window_size(dims, strides, padding):
    def rescale(outputs, inputs, spec):
        if spec is None:
            non_spatial_axes = 0, inputs.ndim - 1
        else:
            non_spatial_axes = spec.index("N"), spec.index("C")

        spatial_shape = tuple(
            inputs.shape[i] for i in range(inputs.ndim) if i not in non_spatial_axes
        )
        one = jnp.ones(spatial_shape, dtype=inputs.dtype)
        window_sizes = lax.reduce_window(one, 0.0, lax.add, dims, strides, padding)
        for i in sorted(non_spatial_axes):
            window_sizes = jnp.expand_dims(window_sizes, i)

        return outputs / window_sizes

    return rescale


AvgPool = _pooling_layer(lax.add, 0.0, _normalize_by_window_size)


def Flatten():
    """Layer construction function for flattening all but the leading dim."""

    def init(rng, input_shape):
        output_shape = input_shape[0], functools.reduce(op.mul, input_shape[1:], 1)
        return output_shape, ()

    def apply(params, inputs, **kwargs):
        return jnp.reshape(inputs, (inputs.shape[0], -1))

    return LayerT(init=init, apply=apply)


Flatten = Flatten()


def Identity():
    """Layer construction function for an identity layer."""
    init = lambda rng, input_shape: (input_shape, ())
    apply = lambda params, inputs, **kwargs: inputs
    return LayerT(init=init, apply=apply)


Identity = Identity()


def FanOut(num):
    """Layer construction function for a fan-out layer."""
    init = lambda rng, input_shape: ([input_shape] * num, ())
    apply = lambda params, inputs, **kwargs: [inputs] * num
    return LayerT(init=init, apply=apply)


def FanInSum():
    """Layer construction function for a fan-in sum layer."""
    init = lambda rng, input_shape: (input_shape[0], ())
    apply = lambda params, inputs, **kwargs: sum(inputs)
    return LayerT(init=init, apply=apply)


FanInSum = FanInSum()


def FanInConcat(axis=-1):
    """Layer construction function for a fan-in concatenation layer."""

    def init(rng, input_shape):
        ax = axis % len(input_shape[0])
        concat_size = sum(shape[ax] for shape in input_shape)
        out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax + 1 :]
        return out_shape, ()

    def apply(params, inputs, **kwargs):
        return jnp.concatenate(inputs, axis)

    return LayerT(init=init, apply=apply)


def Dropout(rate, mode="train"):
    """Layer construction function for a dropout layer with given rate."""

    def init(rng, input_shape):
        return input_shape, ()

    def apply(params, inputs, **kwargs):
        if (rng := kwargs.get("rng", None)) is None:
            msg = (
                "Dropout layer requires `apply` to be called with a PRNG key argument. That is, "
                "instead of `apply(params, inputs)`, call it like `apply(params, inputs, rng)` "
                "where `rng` is a jax.random.PRNGKey value."
            )
            raise ValueError(msg)
        if mode == "train":
            keep = random.bernoulli(rng, rate, inputs.shape)
            return jnp.where(keep, inputs / rate, 0)
        else:
            return inputs

    return LayerT(init=init, apply=apply)


# Composing layers via combinators


def serial(*layers: Sequence[LayerT], verbose: bool = False):
    """Combinator for composing layers in serial.

    Args:
      *layers: a sequence of `LayerT` objects.

    Returns:
      A new `LayerT` object, representing the serial composition of the given sequence of layers.
    """

    nlayers = len(layers)

    def init(rng, input_shape):
        params = []

        if verbose:
            logging.info("Starting with input shape: %s", str(input_shape))

        for lidx, layer in enumerate(layers):
            rng, layer_rng = random.split(rng)
            input_shape, param = layer.init(layer_rng, input_shape)
            params.append(param)
            if verbose:
                logging.info("Output shape in layer %d/%d: %s", lidx + 1, nlayers, str(input_shape))

        return input_shape, params

    def apply(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for layer, param, rng in zip(layers, params, rngs):
            inputs = layer.apply(param, inputs, rng=rng, **kwargs)
        return inputs

    return LayerT(init=init, apply=apply)


def parallel(*layers: Sequence[LayerT]):
    """Combinator for composing layers in parallel.

    The layer resulting from this combinator is often used with the FanOut and
    FanInSum layers.

    Args:
      *layers: a sequence of `LayerT` objexts.

    Returns:
      A `LayerT` object representing the parallel composition of the given sequence of layers. In
      particular, the returned layer takes a sequence of inputs and returns a sequence of outputs
      with the same length as the argument `layers`.
    """
    nlayers = len(layers)

    def init(rng, input_shape):
        rngs = random.split(rng, nlayers)
        return zip(*[l.init(rng, shape) for l, rng, shape in zip(layers, rngs, input_shape)])

    def apply(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        return [l.apply(p, x, rng=r, **kwargs) for l, p, x, r in zip(layers, params, inputs, rngs)]

    return LayerT(init=init, apply=apply)


def shape_dependent(make_layer):
    """Combinator to delay layer constructor pair until input shapes are known.

    Args:
      make_layer: a one-argument function that takes an input shape as an argument
        (a tuple of positive integers) and returns `LayerT` object.

    Returns:
      A `LayerT` object representing the same layer as returned by `make_layer` but with its
      construction delayed until input shapes are known.
    """

    def init(rng, input_shape):
        return make_layer(input_shape)[0](rng, input_shape)

    def apply(params, inputs, **kwargs):
        return make_layer(inputs.shape)[1](params, inputs, **kwargs)

    return LayerT(init=init, apply=apply)
