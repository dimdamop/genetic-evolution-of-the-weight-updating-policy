from collections.abc import Iterable
from typing import Callable

import numpy as np
import jax
from jax import numpy as jnp
import haiku as hk


class Linear(hk.Module):
    """Linear module."""

    def __init__(
        self,
        output_size: int,
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        name: str | None = None,
    ):
        """Constructs the Linear module.

        Args:
            output_size: Output dimensionality.

            w_init: Optional initializer for weights. By default, uses random values from truncated
                normal, with stddev ``1 / sqrt(fan_in)``. See https://arxiv.org/abs/1502.03167v3.

            b_init: Optional initializer for bias. By default, zero.

            name: Name of the module.
        """
        super().__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Computes a linear transform of the input."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")

        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

        out = jnp.dot(inputs, w)

        b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
        b = jnp.broadcast_to(b, out.shape)
        out = out + b

        return out


class MLP(hk.Module):
    """A multi-layer perceptron module."""

    def __init__(
        self,
        output_sizes: Iterable[int],
        w_init: hk.initializers.Initializer | None = None,
        b_init: hk.initializers.Initializer | None = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        activate_final: bool = False,
        name: str | None = None,
    ):
        """Constructs an MLP.

        Args:
            output_sizes: Sequence of layer sizes.

            w_init: Initializer for :class:`~haiku.Linear` weights.

            b_init: Initializer for :class:`~haiku.Linear` bias..

            activation: Activation function to apply between :class:`~haiku.Linear`
                layers. Defaults to ReLU.

            activate_final: Whether or not to activate the final layer of the MLP.

            name: Optional name for this module.
        """

        super().__init__(name=name)
        self.w_init = w_init
        self.b_init = b_init
        self.activation = activation
        self.activate_final = activate_final
        layers = []
        output_sizes = tuple(output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(
                Linear(
                    output_size=output_size,
                    w_init=w_init,
                    b_init=b_init,
                    name="linear_%d" % index,
                )
            )
        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

    def __call__(
        self,
        inputs: jax.Array,
        dropout_rate: float | None = None,
        rng=None,
    ) -> jax.Array:
        """Connects the module to some inputs.

        Args:
            inputs: A Tensor of shape ``[batch_size, input_size]``.

            dropout_rate: Optional dropout rate.

            rng: Optional RNG key. Require when using dropout.

        Returns:
            The output of the model of size ``[batch_size, output_size]``.
        """
        if dropout_rate is not None and rng is None:
            raise ValueError("When using dropout an rng key must be passed.")
        elif dropout_rate is None and rng is not None:
            raise ValueError("RNG should only be passed when using dropout.")

        rng = hk.PRNGSequence(rng) if rng is not None else None
        num_layers = len(self.layers)

        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                # Only perform dropout if we are activating the output.
                if dropout_rate is not None:
                    out = hk.dropout(next(rng), dropout_rate, out)
                out = self.activation(out)

        return out

    def reverse(
        self,
        activate_final: bool | None = None,
        name: str | None = None,
    ) -> "MLP":
        """Returns a new MLP which is the layer-wise reverse of this MLP.

        NOTE: Since computing the reverse of an MLP requires knowing the input size
        of each linear layer this method will fail if the module has not been called
        at least once.

        The contract of reverse is that the reversed module will accept the output
        of the parent module as input and produce an output which is the input size
        of the parent.

        >>> mlp = hk.nets.MLP([1, 2, 3])
        >>> mlp_in = jnp.ones([1, 2])
        >>> y = mlp(mlp_in)
        >>> rev = mlp.reverse()
        >>> rev_mlp_out = rev(y)
        >>> mlp_in.shape == rev_mlp_out.shape
        True

        Args:
            activate_final: Whether the final layer of the MLP should be activated.

            name: Optional name for the new module. The default name will be the name
                of the current module prefixed with ``"reversed_"``.

        Returns:
            An MLP instance which is the reverse of the current instance. Note these
            instances do not share weights and, apart from being symmetric to each
            other, are not coupled in any way.
        """

        if activate_final is None:
            activate_final = self.activate_final
        if name is None:
            name = self.name + "_reversed"

        output_sizes = tuple(
            layer.input_size for layer in reversed(self.layers) if layer.input_size is not None
        )
        if len(output_sizes) != len(self.layers):
            raise ValueError("You cannot reverse an MLP until it has been called.")
        return MLP(
            output_sizes=output_sizes,
            w_init=self.w_init,
            b_init=self.b_init,
            activation=self.activation,
            activate_final=activate_final,
            name=name,
        )
