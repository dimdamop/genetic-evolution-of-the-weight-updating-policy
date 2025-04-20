import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import orbax.checkpoint
from flax.linen.initializers import constant, orthogonal
from flax.training import orbax_utils
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
import distrax
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
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


# checkpointer = orbax.checkpoint.PyTreeCheckpointer()
# save_args = orbax_utils.save_args_from_target(ckpt)
# orbax_checkpointer.save('/tmp/flax_ckpt/orbax/single_save', ckpt, save_args=save_args)


CONFIG = {
    "LR": 3e-4,
    "NUM_ENVS": 2048,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 5e7,
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
    "DEBUG": True,
}

CONFIG["NUM_UPDATES"] = CONFIG["TOTAL_TIMESTEPS"] // CONFIG["NUM_STEPS"] // CONFIG["NUM_ENVS"]
CONFIG["MINIBATCH_SIZE"] = CONFIG["NUM_ENVS"] * CONFIG["NUM_STEPS"] // CONFIG["NUM_MINIBATCHES"]


def linear_schedule(count):
    frac = (
        1.0
        - (count // (CONFIG["NUM_MINIBATCHES"] * CONFIG["UPDATE_EPOCHS"]))
        / CONFIG["NUM_UPDATES"]
    )
    return CONFIG["LR"] * frac


if CONFIG["ANNEAL_LR"]:
    TX = optax.chain(
        optax.clip_by_global_norm(CONFIG["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=LINEAR_SCHEDULE, eps=1e-5),
    )
else:
    TX = optax.chain(
        optax.clip_by_global_norm(CONFIG["MAX_GRAD_NORM"]),
        optax.adam(CONFIG["LR"], eps=1e-5),
    )

ENV, ENV_PARAMS = BraxGymnaxWrapper(CONFIG["ENV_NAME"]), None
ENV = LogWrapper(ENV)
ENV = ClipAction(ENV)
ENV = VecEnv(ENV)

if CONFIG["NORMALIZE_ENV"]:
    ENV = NormalizeVecObservation(ENV)
    ENV = NormalizeVecReward(ENV, CONFIG["GAMMA"])


NETWORK: ActorCritic = ActorCritic(
    ENV.action_space(ENV_PARAMS).shape[0], activation=CONFIG["ACTIVATION"]
)

RNG, _RNG = jax.random.split(jax.random.PRNGKey(12))
NETWORK_PARAMS = NETWORK.init(_RNG, jnp.zeros(ENV.observation_space(ENV_PARAMS).shape))


TRAIN_STATE: TrainState = TrainState.create(
    apply_fn=NETWORK.apply,
    params=NETWORK_PARAMS,
    tx=TX,
)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    def train(rng):
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, CONFIG["NUM_ENVS"])
        obsv, env_state = ENV.reset(reset_rng, ENV_PARAMS)

        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = NETWORK.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, CONFIG["NUM_ENVS"])
                obsv, env_state, reward, done, info = ENV.step(
                    rng_step, env_state, action, ENV_PARAMS
                )
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, CONFIG["NUM_STEPS"]
            )

            train_state, env_state, last_obs, rng = runner_state
            _, last_val = NETWORK.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + CONFIG["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + CONFIG["GAMMA"] * CONFIG["GAE_LAMBDA"] * (1 - done) * gae
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
                        pi, value = NETWORK.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -CONFIG["CLIP_EPS"], CONFIG["CLIP_EPS"]
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
                                1.0 - CONFIG["CLIP_EPS"],
                                1.0 + CONFIG["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + CONFIG["VF_COEF"] * value_loss
                            - CONFIG["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = CONFIG["MINIBATCH_SIZE"] * CONFIG["NUM_MINIBATCHES"]
                assert (
                    batch_size == CONFIG["NUM_STEPS"] * CONFIG["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [CONFIG["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, CONFIG["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * CONFIG["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (TRAIN_STATE, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, CONFIG["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


def main():
    train_jit = jax.jit(make_train(CONFIG))
    out = train_jit(RNG)


if __name__ == "__main__":
    main()
