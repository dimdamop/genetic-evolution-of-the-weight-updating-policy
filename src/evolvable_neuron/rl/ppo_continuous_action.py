import argparse
import json
import os
from datetime import datetime
from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
from etils import epath
from flax.struct import dataclass
from flax.training import orbax_utils
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from tqdm import trange
from wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    LogWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    NormalizeVecRewEnvState,
    VecEnv,
)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@dataclass
class RunnerState:
    train: TrainState
    env: NormalizeVecRewEnvState
    obsv: jnp.ndarray
    rng: jnp.ndarray


def make_train(conf):
    conf["NUM_UPDATES"] = int(conf["TOTAL_TIMESTEPS"] // conf["NUM_STEPS"] // conf["NUM_ENVS"])
    conf["MINIBATCH_SIZE"] = int(conf["NUM_ENVS"] * conf["NUM_STEPS"] // conf["NUM_MINIBATCHES"])
    updates_per_chunk = np.ceil(conf["NUM_UPDATES"] / conf["NUM_CHUNKS"]).astype(int)

    env, env_params = BraxGymnaxWrapper(conf["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if conf["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, conf["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0 - (count // (conf["NUM_MINIBATCHES"] * conf["UPDATE_EPOCHS"])) / conf["NUM_UPDATES"]
        )
        return conf["LR"] * frac

    def train_init(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).shape[0], activation=conf["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if conf["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(conf["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(conf["MAX_GRAD_NORM"]),
                optax.adam(conf["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, conf["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        return RunnerState(train=train_state, env=env_state, obsv=obsv, rng=rng)

    def train_chunk(runner_state: RunnerState):

        network = ActorCritic(env.action_space(env_params).shape[0], activation=conf["ACTIVATION"])

        # TRAIN LOOP
        def _update_step(runner_state: RunnerState, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RunnerState, unused):
                train_state = runner_state.train
                env_state = runner_state.env
                last_obs = runner_state.obsv
                rng = runner_state.rng

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, conf["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                return RunnerState(train=train_state, env=env_state, obsv=obsv, rng=rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, conf["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state = runner_state.train
            env_state = runner_state.env
            last_obs = runner_state.obsv
            rng = runner_state.rng
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + conf["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + conf["GAMMA"] * conf["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -conf["CLIP_EPS"], conf["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - conf["CLIP_EPS"],
                                1.0 + conf["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor + conf["VF_COEF"] * value_loss - conf["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = conf["MINIBATCH_SIZE"] * conf["NUM_MINIBATCHES"]
                assert (
                    batch_size == conf["NUM_STEPS"] * conf["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = traj_batch, advantages, targets
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [conf["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = train_state, traj_batch, advantages, targets, rng
                return update_state, total_loss

            update_state = train_state, traj_batch, advantages, targets, rng
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, conf["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            return RunnerState(train=train_state, env=env_state, obsv=last_obs, rng=rng), metric

        return jax.lax.scan(_update_step, runner_state, None, updates_per_chunk)

    return train_init, train_chunk


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-filepath")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--out-metrics-filepath")
    parser.add_argument("--save-checkpoints-dirpath")
    parser.add_argument("--load-checkpoint-filepath")
    args = parser.parse_args()

    conf = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 1e8,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "hopper",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "NUM_CHUNKS": 3,
    }

    if args.conf_filepath:
        with open(args.conf_filepath, "r") as fd:
            conf.update(json.load(fd))

    return args, conf


def main():

    args, conf = cli()

    train_init, train_chunk = make_train(conf)
    rng = jax.random.PRNGKey(args.random_seed)
    runner_state = jax.jit(train_init)(rng)

    if args.load_checkpoint_filepath or args.save_checkpoints_dirpath:
        ckpter = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    if args.load_checkpoint_filepath:
        runner_state_dict = ckpter.restore(args.load_checkpoint_filepath)["runner_state"]
        # The following doesn't work
        runner_state.train.params = runner_state_dict["train"]["params"]
        runner_state.env = runner_state_dict["env"]
        runner_state.obsv = runner_state_dict["obsv"]
        runner_state.rng = runner_state_dict["rng"]

    train_chunk_jit = jax.jit(train_chunk)

    if args.save_checkpoints_dirpath:
        ckpt_dir = epath.Path(args.save_checkpoints_dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.out_metrics_filepath:
        reported_metrics = {"global_step": [], "chunk": [], "episodic_return": []}

    for chunk in trange(conf["NUM_CHUNKS"]):
        runner_state, metrics = train_chunk_jit(runner_state)
        if args.out_metrics_filepath:
            return_values = metrics["returned_episode_returns"][metrics["returned_episode"]]
            timesteps = metrics["timestep"][metrics["returned_episode"]] * conf["NUM_ENVS"]
            reported_metrics["global_step"] += timesteps.tolist()
            reported_metrics["episodic_return"] += return_values.tolist()
            reported_metrics["chunk"] += [chunk] * len(timesteps)

        if args.save_checkpoints_dirpath:
            timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
            ckpt_path = ckpt_dir / f"{chunk + 1}-of-{conf['NUM_CHUNKS']}_{timestamp}"
            ckpt = {"runner_state": runner_state}
            ckpter.save(ckpt_path, ckpt, save_args=orbax_utils.save_args_from_target(ckpt))

    if args.out_metrics_filepath:
        pd.DataFrame.from_dict(reported_metrics).to_csv(args.out_metrics_filepath, index=False)

    if args.save_checkpoints_dirpath:
        ckpter.wait_until_finished()


if __name__ == "__main__":
    main()
