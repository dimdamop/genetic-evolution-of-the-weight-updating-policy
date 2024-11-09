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
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(conf):

    num_updates = conf["len"]["TOTAL_TIMESTEPS"] // conf["len"]["STEPS"] // conf["len"]["ENVS"]
    minibatch_size = conf["len"]["ENVS"] * conf["len"]["STEPS"] // conf["len"]["MINIBATCHES"]
    env, env_params = BraxGymnaxWrapper(conf["env"]["NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)

    if conf["env"]["NORMALIZE"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, conf["learn"]["GAMMA"])

    def linear_schedule(count):
        frac = 1.0 - (count // (conf["len"]["MINIBATCHES"] * conf["len"]["EPOCHS"])) / num_updates
        return conf["learn"]["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(env.action_space(env_params).shape[0], activation=conf["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, jnp.zeros(env.observation_space(env_params).shape))
        if conf["learn"]["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(conf["learn"]["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(conf["learn"]["MAX_GRAD_NORM"]),
                optax.adam(conf["learn"]["LR"], eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, conf["len"]["ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, conf["len"]["ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, conf["len"]["STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

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
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
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
                batch_size = minibatch_size * conf["len"]["MINIBATCHES"]
                assert (
                    batch_size == conf["len"]["STEPS"] * conf["len"]["ENVS"]
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
                    lambda x: jnp.reshape(x, [conf["len"]["MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
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

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":

    length_conf = {
        # Number of actors
        "ENVS": 2048,
        # Number of steps to let a single actor act on an environment per computation of the
        # advantages the target values for the value network.
        "STEPS": 10,
        # Total number of actor/environment interactions per epoch (across actors)
        "TOTAL_TIMESTEPS": 5e7,
        "EPOCHS": 4,
        "MINIBATCHES": 32,
    }

    learn_conf = {
        # Maximum learning rate
        "LR": 3e-4,
        # Whether to linearly ramp up the learning rate till it reaches `conf["learn"]["LR"]`
        "ANNEAL_LR": False,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
    }

    env_conf = {"NAME": "hopper", "NORMALIZE": True}

    conf = {
        "ACTIVATION": "tanh",
        "DEBUG": 1,
        "len": length_conf,
        "learn": learn_conf,
        "env": env_conf,
    }

    rng = jax.random.PRNGKey(12)
    trainer = make_train(conf)

    if conf["DEBUG"] <= 1:
        train_jit = jax.jit(make_train(conf))
    out = train_jit(rng)
