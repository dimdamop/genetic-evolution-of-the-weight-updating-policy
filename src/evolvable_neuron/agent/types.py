from typing import Any, Callable, Dict, Tuple, NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from jumanji.types import TimeStep


class StepType(jnp.int8):
    """Defines the status of a `TimeStep` within a sequence.

    First: 0
    Mid: 1
    Last: 2
    """

    # Denotes the first `TimeStep` in a sequence.
    FIRST = jnp.array(0, jnp.int8)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = jnp.array(1, jnp.int8)
    # Denotes the last `TimeStep` in a sequence.
    LAST = jnp.array(2, jnp.int8)


class Transition(NamedTuple):
    """Container for a transition."""

    observation: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    discount: chex.ArrayTree
    next_observation: chex.ArrayTree
    log_prob: chex.ArrayTree
    logits: chex.ArrayTree
    extras: Dict | None


class ActorCriticParams(NamedTuple):
    actor: hk.Params
    critic: hk.Params


class ParamsState(NamedTuple):
    """Container for the variables used during the training of an agent."""

    params: ActorCriticParams
    opt_state: optax.OptState
    update_count: float


class ActingState(NamedTuple):
    """Container for data used during the acting in the environment."""

    state: Any
    timestep: TimeStep
    key: chex.PRNGKey
    episode_count: float
    env_step_count: float


class TrainingState(NamedTuple):
    """Container for data used during the training of an agent acting in an environment."""

    params_state: ParamsState | None
    acting_state: ActingState


class Agent:
    """Anakin agent."""

    def __init__(self, total_batch_size: int):

        num_devices = jax.local_device_count()

        if total_batch_size % num_devices != 0:
            raise ValueError(
                f"The total batch size must be a multiple of the number of devices, "
                f"got total_batch_size={total_batch_size} and num_devices={num_devices}."
            )

        self.total_batch_size = total_batch_size
        self.batch_size_per_device = total_batch_size // num_devices

    def init_params(self, key: chex.PRNGKey) -> ParamsState:
        raise NotImplementedError()

    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        raise NotImplementedError()

    def make_policy(self, policy_params: hk.Params | None, stochastic: bool = True) -> Callable:
        raise NotImplementedError()
