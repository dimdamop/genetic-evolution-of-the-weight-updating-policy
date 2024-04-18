# Adapted :mod:`jumanji.training.train`
import inspect
import logging

from contextlib import AbstractContextManager
from functools import partial
from time import perf_counter, time
from typing import Any, Dict, Tuple, Literal

import hydra
import jax
import jax.numpy as jnp
import jumanji as jum
import omegaconf

from chex import PRNGKey
from hydra.utils import instantiate
from jumanji.wrappers import VmapAutoResetWrapper
from jumanji.training.agents.base import Agent
from jumanji.training.types import ActingState, TrainingState


def first_from_device(tree):
    squeeze_fn = lambda x: x[0] if isinstance(x, jnp.ndarray) else x
    return jax.tree_util.tree_map(squeeze_fn, tree)


@hydra.main(config_path="cfg", config_name="rubiks_cube")
def main(cfg: omegaconf.DictConfig, log_compiles: bool = False) -> None:

    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    logging.info({"devices": jax.local_devices()})

    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    logger = instantiate(cfg.logger)
    env = VmapAutoResetWrapper(jum.make(cfg.environment_id))
    agent = instantiate(cfg.agent)(env)
    training_state = _training_state(env, agent, init_key)

    epoch_steps = (
        cfg.training.n_steps
        * cfg.training.total_batch_size
        * cfg.training.num_learner_steps_per_epoch
    )
    eval_timer = Timer(out_var_name="metrics")

    evaluators = [
        instantiate(evlr)(
            eval_env=jum.make(cfg.environment_id),
            agent=agent,
        )
        for evlr in cfg.evaluators
    ]

    @partial(jax.pmap, axis_name="devices")
    def epoch_fn(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        training_state, metrics = jax.lax.scan(
            lambda training_state, _: agent.run_epoch(training_state),
            training_state,
            None,
            cfg.training.num_learner_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, metrics

    logging.info("Starting training")

    with jax.log_compiles(log_compiles), logger:

        epi = 1
        start_t = time()
        last_epoch_t = start_t

        while (train_t := (time() - start_t)) < cfg.training.time_budget:
            key, *evltr_keys = jax.random.split(key, len(evaluators) + 1)

            for evltr_i, (evltr, evl_key) in enumerate(zip(evaluators, evltr_keys)):
                with eval_timer:
                    metrics = evltr.run_evaluation(training_state.params_state, evl_key)
                    jax.block_until_ready(metrics)
                logger.write(
                    data=first_from_device(metrics),
                    label=f"eval/{evltr_i}",
                    env_steps=epi * epoch_steps,
                )

            # training
            with Timer(out_var_name="metrics", num_steps_per_timing=epoch_steps):
                training_state, metrics = epoch_fn(training_state)
                jax.block_until_ready((training_state, metrics))

            logger.write(
                data=first_from_device(metrics),
                label="train",
                env_steps=epi * epoch_steps,
            )

            logging.info(
                "The %d%s epoch finished in %f sec. Remaining time budget: %f sec",
                epi,
                "st" if epi == 1 else "nd" if epi == 2 else "rd" if epi == 3 else "th",
                train_t - last_epoch_t,
                cfg.training.time_budget - train_t,
            )

            last_epoch_t = train_t
            epi += 1

        logging.info(
            "The training lasted %f seconds (with a time budget of %f)",
            train_t,
            cfg.training.time_budget,
        )


def _training_state(env: jum.Environment, agent: Agent, key: PRNGKey) -> TrainingState:
    params_key, reset_key, acting_key = jax.random.split(key, 3)

    # Initialize params
    params_state = agent.init_params(params_key)

    # Initialize environment states
    num_local_devices = jax.local_device_count()
    num_global_devices = jax.device_count()
    num_workers = num_global_devices // num_local_devices
    local_batch_size = agent.total_batch_size // num_global_devices
    reset_keys = jax.random.split(reset_key, agent.total_batch_size).reshape(
        (num_workers, num_local_devices, local_batch_size, -1)
    )
    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(reset_keys[jax.process_index()])

    # Initialize acting states
    acting_key_per_device = jax.random.split(acting_key, num_global_devices).reshape(
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
    def __init__(
        self,
        out_var_name: str | None = None,
        num_steps_per_timing: int | None = None,
    ):
        """Wraps some computation as a context manager. Expects the variable
        `out_var_name` to be newly created within the context of Timer and will
        append some timing metrics to it.

        Args:
            out_var_name: name of the variable to append timing metrics to.
            num_steps_per_timing: number of steps computed during the timing.
        """
        self.out_var_name = out_var_name
        self.num_steps_per_timing = num_steps_per_timing

    def _get_variables(self) -> Dict:
        """Returns the local variables that are accessible in the context of the
        context manager.  This function gets the locals 2 callstacks above.
        Index 0 is this very function, 1 is the __init__/__exit__ level, 2 is
        the context manager level.
        """
        return {(k, id(v)): v for k, v in inspect.stack()[2].frame.f_locals.items()}

    def __enter__(self):
        self._variables_enter = self._get_variables()
        self._start_time = perf_counter()
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        elapsed_time = perf_counter() - self._start_time
        self._variables_exit = self._get_variables()
        self.data = {"time": elapsed_time}
        if self.num_steps_per_timing is not None:
            self.data.update(steps_per_second=int(self.num_steps_per_timing / elapsed_time))
        self._write_in_variable(self.data)
        return False

    def _write_in_variable(self, data: Dict[str, float]) -> None:
        in_context_variables = dict(set(self._variables_exit).difference(self._variables_enter))
        metrics_id = in_context_variables.get(self.out_var_name, None)
        if metrics_id is None:
            logging.debug(
                "Timer did not find variable %s at the context manager level.", self.out_var_name
            )
        else:
            self._variables_exit[("metrics", metrics_id)].update(data)


if __name__ == "__main__":
    main()
