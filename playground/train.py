from functools import partial
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.scipy.special import logsumexp
from jax import grad, vmap, numpy as jnp
from environment.ds_iter import batch_iterator
from learning.stax import Conv, Dense, MaxPool, Relu, Flatten, Softmax, serial
from learning.optimizers import momentum, constant


def one_hot(x, k: int = 1, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def binary_cross_entropy(y_true, y_pred, tol=1e-6):
    return y_true * jnp.log(y_pred + tol) + (1 - y_true) * jnp.log(1 - y_pred + tol)


def logistic_loss(params, predictor, feats, targets):
    preds = vmap(partial(predictor, params))(feats)
    bces = vmap(binary_cross_entropy)(targets, preds)
    return -jnp.sum(bces)


def main():
    rng = random.PRNGKey(0)

    # Prevent tf from grabbing all GPU memory
    tf.config.set_visible_devices([], device_type="GPU")

    sgd_step_size = 1e-4
    sgd_momentum = 0.9
    num_opt_iters = 1000
    report_every = 50
    batch_size = 16
    img_dim_len = 96

    ds_gen = batch_iterator(
        resize_to=[img_dim_len, img_dim_len],
        batch_size=batch_size,
        return_mask=False,
        return_regression_tgt=False,
    )

    loss = grad(logistic_loss)

    net_init, net_apply = serial(
        Conv(32, (3, 3), padding="SAME"),
        Relu,
        Conv(64, (3, 3), padding="SAME"),
        Relu,
        MaxPool((2, 2)),
        Flatten,
        Conv(32, (3, 3), padding="SAME"),
        Relu,
        Conv(64, (3, 3), padding="SAME"),
        Relu,
        Dense(128),
        Relu,
        Dense(2),
        Softmax,
    )
    in_shape = (-1, img_dim_len, img_dim_len, 3)
    out_shape, net_params = net_init(rng, in_shape)

    opt_init, opt_update, opt_get_params = momentum(
        step_size=constant(sgd_step_size),
        mass=sgd_momentum,
    )
    opt_update = jit(opt_update)
    opt_get_params = jit(opt_get_params)
    opt_state = opt_init(net_params)

    @jit
    def opt_step(opt_iter: int, opt_state, batch):
        opt_params = opt_get_params(opt_state)
        loss, gradients = jax.value_and_grad(logistic_loss)(opt_params, batch)
        return loss, opt_update(opt_iter, gradients, opt_state)

    for opt_iter, train_batch in enumerate(ds_gen):
        if opt_iter == num_opt_iters:
            break

        net_eval, net_grads = loss_value_and_grad(opt_get_params(opt_state))
        opt_state = opt_update(opt_iter, net_grads, opt_state)

        start_time = time.time()
        epoch_time = time.time() - start_time

        if opt_iter % report_every == 0:
            end_time = time.time()
            print(f"Iteration {opt_iter + 1}/{num_opt_iters} ({end_time - start_time:0.2f} s)")
            start_time = end_time
            test_batch = next(ds_gen)
            print(f"Accuracy: {accuracy(params, test_batch)}")


if __name__ == "__main__":
    main()
