# Copied from :mod:`jumanji.training.evaluator`
import functools
from typing import Any, Dict, Tuple

import chex
import haiku as hk
import jax

from jax.random import split as split_key
from jax import numpy as jnp

from jumanji.env import Environment
from .types import Agent, ActingState, ParamsState


class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: Environment, agent: Agent,
        total_batch_size: int,
        stochastic: bool,
    ):
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
            functools.partial(self._gen_evals, batch_size=self.batch_size_per_device),
            axis_name="devices",
        )
        self.stochastic = stochastic

    def _eval_one_episode(self, policy_params: hk.Params | None, key: chex.PRNGKey) -> Dict:
        policy = self.agent.make_policy(policy_params=policy_params, stochastic=self.stochastic)

        def acting_policy(observation: Any, key: chex.PRNGKey) -> chex.Array:
            action, _ = policy(observation, key)
            return action

        def cond_fun(carry: Tuple[ActingState, float]) -> jnp.bool_:
            acting_state, _ = carry
            return ~acting_state.timestep.last()

        def body_fun(carry: Tuple[ActingState, float]) -> Tuple[ActingState, float]:
            acting_state, return_ = carry
            key, action_key = split_key(acting_state.key)
            observation = jax.tree_util.tree_map(
                lambda x: x[None], acting_state.timestep.observation
            )
            action = acting_policy(observation, action_key)
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

        reset_key, init_key = split_key(key)
        state, timestep = self.eval_env.reset(reset_key)
        acting_state = ActingState(
            state=state,
            timestep=timestep,
            key=init_key,
            episode_count=jnp.array(0, jnp.int32),
            env_step_count=jnp.array(0, jnp.int32),
        )
        return_ = jnp.array(0, float)
        final_acting_state, return_ = jax.lax.while_loop(
            cond_fun, body_fun, (acting_state, return_),
        )
        eval_metrics = {
            "episode_return": return_,
            "episode_length": final_acting_state.env_step_count,
        }
        if extras := final_acting_state.timestep.extras:
            eval_metrics.update(extras)
        return eval_metrics

    def _gen_evals(self, params_state: ParamsState, key: chex.PRNGKey, batch_size: int) -> Dict:
        policy_params = params_state.params.actor
        keys = split_key(key, batch_size)
        eval_metrics = jax.vmap(self._eval_one_episode, in_axes=(None, 0))(policy_params, keys)
        return jax.lax.pmean(jax.tree_util.tree_map(jnp.mean, eval_metrics), axis_name="devices")

    def run_evaluation(self, params_state: ParamsState | None, eval_key: chex.PRNGKey) -> Dict:
        """Run one batch of evaluations."""
        eval_keys = split_key(eval_key, self.num_global_devices).reshape(
            self.num_workers, self.num_local_devices, -1
        )
        return self.generate_evaluations(params_state, eval_keys[jax.process_index()])
