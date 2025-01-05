import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from evolvable_neuron.rl.wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    NormalizeVecRewEnvState,
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


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: NormalizeVecRewEnvState
    last_obs: jax.Array
    rng: jax.Array


def env_step(runner_state, env, env_params, num_actors: int):
    """collect trajectories"""

    train_state, env_state, last_obs, rng = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi, value = train_state.apply_fn(train_state.params, last_obs)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, num_actors)
    obsv, env_state, reward, done, info = env.step(
        rng_step, env_state, action, env_params
    )
    transition = Transition(done, action, value, reward, log_prob, last_obs, info)
    runner_state = train_state, env_state, obsv, rng

    return runner_state, transition


def makeenv(name: str, normalize: bool, gamma: float):
    env = BraxGymnaxWrapper(name)
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)

    if normalize:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, gamma)
    return env, None


def make_train(conf):

    env, env_params = makeenv(conf["env"]["NAME"], conf["env"]["NORMALIZE"], conf["learn"]["GAMMA"])

    def train(rng, train_state: TrainState, num_updates: int):
        # TRAIN LOOP
        def _update_step(runner_state, unused):

            def _env_step(runner_state, unused):
                return env_step(
                    runner_state=runner_state,
                    env=env,
                    env_params=env_params,
                    num_actors=conf["len"]["ENVS"],
                )

            # COLLECT TRAJECTORIES
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, conf["len"]["STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = train_state.apply_fn(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + conf["learn"]["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + conf["learn"]["GAMMA"] * conf["learn"]["GAE_LAMBDA"] * (1 - done) * gae
                    )
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
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = train_state.apply_fn(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -conf["learn"]["CLIP_EPS"], conf["learn"]["CLIP_EPS"]
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
                                1.0 - conf["learn"]["CLIP_EPS"],
                                1.0 + conf["learn"]["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + conf["learn"]["VF_COEF"] * value_loss
                            - conf["learn"]["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = conf["len"]["STEPS"] * conf["len"]["ENVS"]
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [conf["len"]["MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, conf["len"]["EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if conf["DEBUG"] > 0:

                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * conf["len"]["ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, conf["len"]["ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        runner_state = train_state, env_state, obsv, rng
        # runner_state, metric = jax.lax.scan(_update_step, runner_state, None, 1_000)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, 1_00)
        return runner_state[0].params

    return train


def main():
    length_conf = {
        # Number of actors to collect trajectories from
        "ENVS": 2048,
        # Number of actor/environment interactions (steps) per epoch, per environment and per actor
        "STEPS": 10,
        # Total number of steps for the entire duration of training
        "TOTAL_TIMESTEPS": 5e7,
        "EPOCHS": 4,
        "MINIBATCHES": 32,
    }

    learn_conf = {
        # Maximum learning rate
        "LR": 3e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
    }

    env_conf = {
        "NAME": "hopper",
        "NORMALIZE": True,
        "action_dim": 3,
        "observation_dim": 11,
    }

    conf = {
        "DEBUG": 1,
        "len": length_conf,
        "learn": learn_conf,
        "env": env_conf,
    }

    rng = jax.random.PRNGKey(23)

    # INIT NETWORK PARAMS
    network = ActorCritic(conf["env"]["action_dim"])
    rng, _rng = jax.random.split(rng)
    network_params = network.init(_rng, jnp.zeros(conf["env"]["observation_dim"]))
    tx = optax.chain(
        optax.clip_by_global_norm(conf["learn"]["MAX_GRAD_NORM"]),
        optax.adam(conf["learn"]["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    trainer = jax.jit(make_train(conf)) if conf["DEBUG"] <= 1 else make_train(conf)

    all_updates = conf["len"]["TOTAL_TIMESTEPS"] // conf["len"]["STEPS"] // conf["len"]["ENVS"]
    cut = 1_000

    for _ in range(0, int(all_updates), cut):
        train_state = trainer(rng=rng, train_state=train_state, num_updates=cut)


if __name__ == "__main__":
    main()
