import logging
import time

import jax.numpy as jnp
import tensorflow as tf
from jax import grad, jit, random

from evolvable_neuron.env.supervised_autogen import batch_iterator
from evolvable_neuron.learning.optimizers import constant as constant_sched
from evolvable_neuron.learning.optimizers import sgd
from evolvable_neuron.learning.stax import (
    BatchNorm,
    Conv,
    Dense,
    Flatten,
    LogSoftmax,
    MaxPool,
    Relu,
    serial,
)


def model():
    return serial(
        Conv(32, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        Conv(64, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        MaxPool((2, 2)),
        Conv(32, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        Conv(64, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        MaxPool((2, 2)),
        Conv(32, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        Conv(64, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        MaxPool((2, 2)),
        Conv(32, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        Conv(64, (3, 3), padding="SAME"),
        BatchNorm((0, 1)),
        Relu,
        MaxPool((12, 12)),
        Flatten,
        Dense(32),
        BatchNorm((0, 1)),
        Relu,
        Dense(32),
        BatchNorm((0, 1)),
        Relu,
        Dense(2),
        LogSoftmax,
    )


def main():
    logging.basicConfig(level=logging.DEBUG)

    rng = random.PRNGKey(0)

    # Prevent tf from grabbing all GPU memory
    tf.config.set_visible_devices([], device_type="GPU")

    sgd_step_size = 1e-5
    max_opt_iters = 10000
    report_every = 10
    batch_size = 64
    img_dim_len = 96

    ds_gen = batch_iterator(resize_to=[img_dim_len, img_dim_len], batch_size=batch_size)

    in_shape = (-1, img_dim_len, img_dim_len, 3)
    net = model()
    _, net_params = net.init(rng, in_shape)

    opt = sgd(step_size=constant_sched(sgd_step_size))
    opt_state = opt.init(net_params)

    @jit
    def loss(params, batch):
        inputs, targets = batch
        preds = net.apply(params, inputs)
        return -(targets * preds).sum(-1).mean()

    @jit
    def accuracy(params, batch):
        inputs, targets = batch
        preds = net.apply(params, inputs)
        hard_targets = jnp.argmax(targets, axis=-1)
        hard_preds = jnp.argmax(preds, axis=-1)
        return (hard_targets == hard_preds).mean()

    @jit
    def opt_step(opt_iter: int, opt_state, batch) -> None:
        params = opt.get_params(opt_state)
        return opt.update(opt_iter, grad(loss)(params, batch), opt_state)

    started_at = time.time()

    for opt_iter, batch in enumerate(ds_gen):
        if opt_iter == max_opt_iters:
            break

        train_batch = batch["img"].numpy(), batch["binary"].numpy()
        opt_state = opt_step(opt_iter, opt_state, train_batch)

        if (opt_iter + 1) % report_every == 0:
            ended_at = time.time()
            epoch_duration = ended_at - started_at
            started_at = ended_at
            curr_loss = loss(opt.get_params(opt_state), train_batch)
            curr_accuracy = accuracy(opt.get_params(opt_state), train_batch)
            print(
                f"epoch {opt_iter + 1}/{max_opt_iters} ({epoch_duration:1.2f} s): "
                f"batch loss: {curr_loss:1.5f}, batch accuracy: {100 * curr_accuracy:2.3} %"
            )


if __name__ == "__main__":
    main()
