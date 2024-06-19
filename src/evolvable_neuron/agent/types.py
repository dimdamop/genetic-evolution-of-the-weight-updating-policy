# Copyright 2022 InstaDeep Ltd. All rights reserved.
# Copyright 2024 Dimitrios Damopoulos. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.  See the License for the specific language governing permissions and limitations under
# the License.
#
# This source code is a copy of `jumanji.training.types` with only very minor modifications. In
# particular:
# - the `haiku.Params` type has been replaced by `FrozenDict[str, Any`]
# - `ActorCriticParams` has been merged to `ParamsState`
# - Various variables have been renamed, such as `TrainingState` getting renamed to `TrainState`
# - More docstrings

from typing import Any, Dict, NamedTuple, Optional

import chex
import optax
from flax.core import FrozenDict
from jumanji.types import TimeStep


class Transition(NamedTuple):
    """Container for a transition."""

    observation: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    discount: chex.ArrayTree
    next_observation: chex.ArrayTree
    log_prob: chex.ArrayTree
    logits: chex.ArrayTree
    extras: Optional[Dict]


class NetworkVariables(NamedTuple):
    backprop: FrozenDict[str, Any]
    memory: FrozenDict[str, Any]


class ParamsState(NamedTuple):
    """Container for the variables used during the training of an agent."""

    actor: NetworkVariables
    critic: NetworkVariables
    opt: optax.OptState

    # Not used anywhere, but the original `jumanji` implementation has it and
    # `flax.training.train_state.TrainState` has it too. I trust that all of these guys have good
    # reason for carrying this around, so -whatever- let' s keep it.
    update_count: float


class ActingState(NamedTuple):
    """Container for data used during the acting in the environment."""

    state: Any
    timestep: TimeStep
    key: chex.PRNGKey
    episode_count: float
    env_step_count: float


class TrainState(NamedTuple):
    """Container for data used during the training of an agent acting in an environment."""

    params: Optional[ParamsState]
    acting: ActingState
