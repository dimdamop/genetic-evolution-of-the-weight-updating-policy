# Adapted :mod:`jumanji.training.train`
import inspect
import logging
import pickle

from contextlib import AbstractContextManager
from functools import partial
from time import perf_counter
from typing import Any, Dict, Tuple, Literal

import hydra
import jax
import jax.numpy as jnp
import jumanji as jum
import omegaconf

from chex import PRNGKey
from hydra.utils import instantiate
from jax.random import split as split_key
from jumanji.wrappers import VmapAutoResetWrapper
from jumanji.training.agents.base import Agent
from jumanji.training.types import ActingState, ActorCriticParams, ParamsState, TrainingState
from jumanji.training.evaluator import Evaluator


def first_from_device(tree):
    squeeze_fn = lambda x: x[0] if isinstance(x, jnp.ndarray) else x
    return jax.tree_util.tree_map(squeeze_fn, tree)


def write_params_state(params_state: ParamsState, path: str) -> None:
    with open(path, "wb") as stream:
        pickle.dump(first_from_device(params_state), stream)


def read_params_state(path: str) -> ParamsState:
    with open(path, "rb") as stream:
        return jax.device_put(pickle.load(stream))


@hydra.main(config_path="cfg", config_name="rubiks_cube")
def main(cfg: omegaconf.DictConfig, log_compiles: bool = False) -> None:

    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    logging.info({"devices": jax.local_devices()})

    train_key = jax.random.PRNGKey(cfg.seed.train)
    train_eval_key, final_eval_key = split_key(jax.random.PRNGKey(cfg.seed.evaluation))

    logger = instantiate(cfg.logger)
    env = VmapAutoResetWrapper(jum.make(cfg.environment_id))
    agent = instantiate(cfg.agent)(env)
    training_state = _training_state(env, agent, train_key, cfg.get("params_path"))

    epoch_steps = (
        cfg.training.envs_in_parallel
        * cfg.training.env_steps_per_update
        * cfg.training.num_learner_updates_per_epoch
    )

    eval_timer = Timer(out_var_name="metrics")
    train_timer = Timer(out_var_name="metrics", num_steps_per_timing=epoch_steps)

    evaluators = {
        k: instantiate(v)(
            eval_env=jum.make(cfg.environment_id),
            agent=agent,
        )
        for k, v in cfg.evaluators.items()
    }

    @partial(jax.pmap, axis_name="devices")
    def epoch_fn(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        training_state, metrics = jax.lax.scan(
            lambda training_state, _: agent.run_epoch(training_state),
            training_state,
            None,
            cfg.training.num_learner_updates_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, metrics

    logging.info("Starting training. The time budget is %.1f sec", cfg.training.time_budget)

    with jax.log_compiles(log_compiles), logger:
        epi, total_steps, last_epoch_t, remaining_t = 0, 0, 0, cfg.training.time_budget

        while remaining_t >= last_epoch_t:

            _train_eval_key, train_eval_key = split_key(train_eval_key)

            for eval_id, metrics in evaluate(
                training_state=training_state,
                evaluators=evaluators,
                key=_train_eval_key,
                timer=eval_timer,
            ).items():
                logger.write(
                    data=first_from_device(metrics),
                    label=f"eval/train/{eval_id}",
                    env_steps=total_steps,
                )

            # training
            with train_timer:
                training_state, metrics = epoch_fn(training_state)
                jax.block_until_ready((training_state, metrics))

            logger.write(data=first_from_device(metrics), label="train", env_steps=total_steps)

            last_epoch_t = train_timer.elapsed_time
            if epi > 0:
                # We gift the first epoch, the duration of which is affected but initialization
                # operations.
                remaining_t -= last_epoch_t
            total_steps += epoch_steps
            epi += 1

            logging.info(
                "The %d%s epoch finished in %.1f sec. Remaining time credit: %.1f sec",
                epi,
                "st" if epi == 1 else "nd" if epi == 2 else "rd" if epi == 3 else "th",
                last_epoch_t,
                remaining_t,
            )

        logging.info(
            "The training lasted %.1f sec (with a time budget of %.1f sec)",
            cfg.training.time_budget - remaining_t,
            cfg.training.time_budget,
        )

        for eval_id, metrics in evaluate(
            training_state=training_state,
            evaluators=evaluators,
            key=final_eval_key,
            timer=eval_timer,
        ).items():
            metrics_path = str(eval_id) + "_evaluation.out"
            logging.info("Storing the metric values in %s...", metrics_path)
            with open(metrics_path, "w") as stream:
                stream.write(str(first_from_device(metrics)))

        params_path = "params_state.pkl"
        logging.info("Storing the parameters of the agent and its training at %s", params_path)
        write_params_state(training_state.params_state, params_path)


def _training_state(
    env: jum.Environment,
    agent: Agent,
    key: PRNGKey,
    params_path: str | None,
) -> TrainingState:

    params_key, reset_key, acting_key = split_key(key, 3)

    # Initialize params
    if params_path:
        logging.info("Loading the parameters of the agent and its training from %s", params_path)
        params_state = read_params_state(params_path)
    else:
        params_state = agent.init_params(params_key)

    # Initialize environment states
    num_local_devices = jax.local_device_count()
    num_global_devices = jax.device_count()
    num_workers = num_global_devices // num_local_devices
    local_batch_size = agent.total_batch_size // num_global_devices
    reset_keys = split_key(reset_key, agent.total_batch_size).reshape(
        (num_workers, num_local_devices, local_batch_size, -1)
    )
    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(reset_keys[jax.process_index()])

    # Initialize acting states
    acting_key_per_device = split_key(acting_key, num_global_devices).reshape(
        num_workers, num_local_devices, -1
    )

    acting_state = ActingState(
        state=env_state,
        timestep=timestep,
        key=acting_key_per_device[jax.process_index()],
        episode_count=jnp.zeros(num_local_devices, float),
        env_step_count=jnp.zeros(num_local_devices, float),
    )

    return TrainingState(
        params_state=jax.device_put_replicated(params_state, jax.local_devices()),
        acting_state=acting_state,
    )


class Timer(AbstractContextManager):
    def __init__(self, out_var_name: str | None = None, num_steps_per_timing: int | None = None):
        """Wraps some computation as a context manager. Expects the variable `out_var_name` to be
        newly created within the context of Timer and will append some timing metrics to it.

        Args:
            out_var_name: name of the variable to append timing metrics to.

            num_steps_per_timing: number of steps computed during the timing.
        """
        self.out_var_name = out_var_name
        self.num_steps_per_timing = num_steps_per_timing

    def _get_variables(self) -> Dict:
        """Returns the local variables that are accessible in the context of the context manager"""

        # We have to ascent 3 callstacks: the first is this very function, the second is the caller
        # (ie., `__init__` or `__exit__`) and the third is the context manager level that we need.
        return {(k, id(v)): v for k, v in inspect.stack()[2].frame.f_locals.items()}

    def __enter__(self):
        self._variables_enter = self._get_variables()
        self._start_time = perf_counter()
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        self.elapsed_time = perf_counter() - self._start_time
        self._variables_exit = self._get_variables()
        self.data = {"time": self.elapsed_time}
        if self.num_steps_per_timing is not None:
            self.data.update(steps_per_second=int(self.num_steps_per_timing / self.elapsed_time))
        self._write_in_variable(self.data)
        return False

    def _write_in_variable(self, data: Dict[str, float]) -> None:
        in_context_variables = dict(set(self._variables_exit).difference(self._variables_enter))
        metrics_id = in_context_variables.get(self.out_var_name)
        if metrics_id is None:
            logging.debug(
                "Timer did not find variable %s at the context manager level.", self.out_var_name
            )
        else:
            self._variables_exit[("metrics", metrics_id)].update(data)


def evaluate(training_state, evaluators: Dict[str, Evaluator], key, timer: Timer):

    all_metrics = {}

    for eval_id, evaluator in evaluators.items():
        eval_key, key = split_key(key)
        with timer:
            metrics = evaluator.run_evaluation(training_state.params_state, eval_key)
            jax.block_until_ready(metrics)
        all_metrics[eval_id] = metrics

    return all_metrics


if __name__ == "__main__":
    main()
