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
# This source code is a copy of `jumanji.training.agents.a2c.a2c_agent` with only very minor
# modifications. In particular:
# - the `haiku.Params` type has been replaced by `FrozenDict[str, Any`]
# - The loss is computed using `TrainState` directly, as opposed to passing down the encapsulated
#   objects.

import functools
from typing import Any, Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
import rlax
from flax import linen as nn
from flax.core import FrozenDict
from jumanji.env import Environment
from jumanji.training.networks.parametric_distribution import ParametricDistribution

from evolvable_neuron.agent.types import ActingState, ParamsState, TrainState, Transition


class Agent:
    def __init__(
        self,
        env: Environment,
        n_steps: int,
        total_batch_size: int,
        policy: nn.Module,
        value: nn.Module,
        parametric_action_distribution: ParametricDistribution,
        optimizer: optax.GradientTransformation,
        normalize_advantage: bool,
        discount_factor: float,
        bootstrapping_factor: float,
        l_pg: float,
        l_td: float,
        l_en: float,
    ) -> None:
        self.total_batch_size = total_batch_size
        num_devices = jax.local_device_count()
        assert total_batch_size % num_devices == 0, (
            "The total batch size must be a multiple of the number of devices, "
            f"got total_batch_size={total_batch_size} and num_devices={num_devices}."
        )

        self.batch_size_per_device = total_batch_size // num_devices
        self.env = env
        self.observation_spec = env.observation_spec
        self.n_steps = n_steps
        self.policy = policy
        self.value = value
        self.parametric_action_distribution = parametric_action_distribution 
        self.optimizer = optimizer
        self.normalize_advantage = normalize_advantage
        self.discount_factor = discount_factor
        self.bootstrapping_factor = bootstrapping_factor
        self.l_pg = l_pg
        self.l_td = l_td
        self.l_en = l_en

    def init_params(self, key: chex.PRNGKey) -> ParamsState:
        actor_key, critic_key = jax.random.split(key)
        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[None, ...], self.observation_spec.generate_value()
        )  # Add batch dim

        actor = self.policy.init(actor_key, dummy_obs)["params"]
        critic = self.value.init(critic_key, dummy_obs)["params"]
        return ParamsState(
            actor=actor,
            critic=critic,
            opt=self.optimizer.init((actor, critic)),
            update_count=jnp.array(0, float),
        )

    def training_iteration(self, train_state: TrainState) -> Tuple[TrainState, Dict]:
        """Performs a single step of parameters' update using the provided optimizer. It returns the
        updated training state and the computed metrics.
        """
        grad, (acting_state, metrics) = jax.grad(self.a2c_loss, has_aux=True)(train_state)
        grad, metrics = jax.lax.pmean((grad, metrics), "devices")
        updates, opt = self.optimizer.update(grad, train_state.params.opt)
        ac, cr = optax.apply_updates((train_state.params.actor, train_state.params.critic), updates)
        train_state = TrainState(
            params=ParamsState(
                actor=ac, critic=cr, opt=opt, update_count=train_state.params.update_count + 1,
            ),
            acting_state=acting_state,
        )
        return train_state, metrics

    def a2c_loss(self, train_state: TrainState) -> Tuple[float, Tuple[ActingState, Dict]]:
        parametric_action_distribution = self.parametric_action_distribution
        value_apply = self.value.apply

        acting_state, data = self.rollout(
            policy_params=train_state.params.actor, acting_state=train_state.acting
        )  # data.shape == (T, B, ...)
        last_observation = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        observation = jax.tree_util.tree_map(
            lambda obs_0_tm1, obs_t: jnp.concatenate([obs_0_tm1, obs_t[None]], axis=0),
            data.observation,
            last_observation,
        )

        value = jax.vmap(value_apply, in_axes=(None, 0))(train_state.params.critic, observation)
        discounts = jnp.asarray(self.discount_factor * data.discount, float)
        value_tm1 = value[:-1]
        value_t = value[1:]
        advantage = jax.vmap(
            functools.partial(
                rlax.td_lambda,
                lambda_=self.bootstrapping_factor,
                stop_target_gradients=True,
            ),
            in_axes=1,
            out_axes=1,
        )(
            value_tm1,
            data.reward,
            discounts,
            value_t,
        )

        # Compute the critic loss before potentially normalizing the advantages.
        critic_loss = jnp.mean(advantage**2)

        # Compute the policy loss with optional advantage normalization.
        metrics: Dict = {}
        if self.normalize_advantage:
            metrics.update(unnormalized_advantage=jnp.mean(advantage))
            advantage = jax.nn.standardize(advantage)
        policy_loss = -jnp.mean(jax.lax.stop_gradient(advantage) * data.log_prob)

        # Compute the entropy loss, i.e. negative of the entropy.
        entropy = jnp.mean(parametric_action_distribution.entropy(data.logits, acting_state.key))
        entropy_loss = -entropy

        total_loss = self.l_pg * policy_loss + self.l_td * critic_loss + self.l_en * entropy_loss

        metrics.update(
            total_loss=total_loss,
            policy_loss=policy_loss,
            critic_loss=critic_loss,
            entropy_loss=entropy_loss,
            entropy=entropy,
            advantage=jnp.mean(advantage),
            value=jnp.mean(value),
        )
        if data.extras:
            metrics.update(data.extras)
        return total_loss, (acting_state, metrics)

    def make_policy(
        self,
        policy_params: FrozenDict[str, Any],
        stochastic: bool = True,
    ) -> Callable[[Any, chex.PRNGKey], Tuple[chex.Array, Tuple[chex.Array, chex.Array]]]:
        policy_network = self.policy
        parametric_action_distribution = self.parametric_action_distribution

        def policy(
            observation: Any, key: chex.PRNGKey
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
            logits = policy_network.apply(policy_params, observation)
            if stochastic:
                raw_action = parametric_action_distribution.sample_no_postprocessing(logits, key)
                log_prob = parametric_action_distribution.log_prob(logits, raw_action)
            else:
                del key
                raw_action = parametric_action_distribution.mode_no_postprocessing(logits)
                # log_prob is log(1), i.e. 0, for a greedy policy (deterministic distribution).
                log_prob = jnp.zeros_like(
                    parametric_action_distribution.log_prob(logits, raw_action)
                )
            action = parametric_action_distribution.postprocess(raw_action)
            return action, (log_prob, logits)

        return policy

    def rollout(
        self,
        policy_params: FrozenDict[str, Any],
        acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """
        policy = self.make_policy(policy_params=policy_params)

        def run_one_step(
            acting_state: ActingState, key: chex.PRNGKey
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            action, (log_prob, logits) = policy(timestep.observation, key)
            next_env_state, next_timestep = self.env.step(acting_state.state, action)

            acting_state = ActingState(
                state=next_env_state,
                timestep=next_timestep,
                key=key,
                episode_count=acting_state.episode_count
                + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count
                + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                next_observation=next_timestep.observation,
                log_prob=log_prob,
                logits=logits,
                extras=next_timestep.extras,
            )

            return acting_state, transition

        acting_keys = jax.random.split(acting_state.key, self.n_steps).reshape((self.n_steps, -1))
        acting_state, data = jax.lax.scan(run_one_step, acting_state, acting_keys)
        return acting_state, data
