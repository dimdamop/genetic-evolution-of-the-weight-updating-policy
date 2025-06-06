# Copyright 2022 InstaDeep Ltd. All rights reserved.
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
#
# This source code is a copy of `jumanji.training.evaluator` with only very minor modifications.

import functools
from typing import Dict, Optional, Tuple

import chex
import jax
from jax import numpy as jnp
from jumanji.env import Environment

from evolvable_neuron.agent.types import ActingState, ParamsState, VarCollection


class Evaluator:
    """Class to run evaluations."""

    def __init__(self, eval_env: Environment, agent, total_batch_size: int, stochastic: bool):
        self.eval_env = eval_env
        self.agent = agent
        self.num_local_devices = jax.local_device_count()
        self.num_global_devices = jax.device_count()
        self.num_workers = self.num_global_devices // self.num_local_devices
        if total_batch_size % self.num_global_devices != 0:
            raise ValueError(
                "Expected eval total_batch_size to be a multiple of num_devices, "
                f"got {total_batch_size} and {self.num_global_devices}."
            )
        self.total_batch_size = total_batch_size
        self.batch_size_per_device = total_batch_size // self.num_global_devices
        self.generate_evaluations = jax.pmap(
            functools.partial(
                self._generate_evaluations, eval_batch_size=self.batch_size_per_device
            ),
            axis_name="devices",
        )
        self.stochastic = stochastic

    def _eval_one_episode(self, policy_params: Optional[VarCollection], key: chex.PRNGKey) -> Dict:
        policy = self.agent.make_policy(policy_params=policy_params, stochastic=self.stochastic)

        def cond_fun(carry: Tuple[ActingState, float]) -> jnp.bool_:
            acting_state, _ = carry
            return ~acting_state.timestep.last()

        def body_fun(
            carry: Tuple[ActingState, float],
        ) -> Tuple[ActingState, float]:
            acting_state, return_ = carry
            key, action_key = jax.random.split(acting_state.key)
            observation = jax.tree_util.tree_map(
                lambda x: x[None], acting_state.timestep.observation
            )
            action, _ = policy(observation, action_key)
            state, timestep = self.eval_env.step(acting_state.state, jnp.squeeze(action, axis=0))
            return_ += timestep.reward
            acting_state = ActingState(
                state=state,
                timestep=timestep,
                key=key,
                episode_count=jnp.array(0, jnp.int32),
                env_step_count=acting_state.env_step_count + 1,
            )
            return acting_state, return_

        reset_key, init_key = jax.random.split(key)
        state, timestep = self.eval_env.reset(reset_key)
        acting_state = ActingState(
            state=state,
            timestep=timestep,
            key=init_key,
            episode_count=jnp.array(0, jnp.int32),
            env_step_count=jnp.array(0, jnp.int32),
        )
        return_ = jnp.array(0, float)
        last_act_state, return_ = jax.lax.while_loop(cond_fun, body_fun, (acting_state, return_))
        eval_metrics = {"episode_return": return_, "episode_length": last_act_state.env_step_count}

        if extras := last_act_state.timestep.extras:
            eval_metrics.update(extras)

        return eval_metrics

    def _generate_evaluations(
        self,
        params_state: ParamsState,
        key: chex.PRNGKey,
        eval_batch_size: int,
    ) -> Dict:
        policy_params = params_state.params.actor
        keys = jax.random.split(key, eval_batch_size)
        eval_metrics = jax.vmap(self._eval_one_episode, in_axes=(None, 0))(
            policy_params,
            keys,
        )
        eval_metrics: Dict = jax.lax.pmean(
            jax.tree_util.tree_map(jnp.mean, eval_metrics),
            axis_name="devices",
        )

        return eval_metrics

    def run_evaluation(self, params_state: Optional[ParamsState], eval_key: chex.PRNGKey) -> Dict:
        """Run one batch of evaluations."""
        eval_keys = jax.random.split(eval_key, self.num_global_devices).reshape(
            self.num_workers, self.num_local_devices, -1
        )
        eval_keys_per_worker = eval_keys[jax.process_index()]
        eval_metrics: Dict = self.generate_evaluations(
            params_state,
            eval_keys_per_worker,
        )
        return eval_metrics
