from typing import Tuple

import flax
import jax
import optax
import pytest

from jax import numpy as jnp
from tqdm import tqdm


@pytest.mark.parametrize(
    ("ds_conf", "layer_feats", "lr"),
    ((((2, 4), 16), [128, 32, 1], 1e-3), (((8, 16), 128), [128, 128, 64, 64, 32, 1], 1e-4)),
)
def test_supervised_regression_with_mlp_with_memory(ds, mlp_with_memory, tx) -> None:

    def update_step(apply_fn, x, y_true, opt_state, params, state):
        def loss(params):
            y_pred, updated_state = apply_fn(
                {"params": params, **state}, x, mutable=list(state.keys())
            )
            l = ((y_pred - y_true) ** 2).mean()
            return l, updated_state

        (l, updated_state), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return opt_state, params, updated_state, l

    batch = next(iter(ds))
    x = batch["img"].numpy().reshape(-1)
    variables = mlp_with_memory.init(jax.random.key(0), x)
    state, params = flax.core.pop(variables, "params")
    del variables
    opt_state = tx.init(params)

    for batch_idx, batch in enumerate(ds):
        opt_state, params, state, loss = update_step(
            apply_fn=mlp_with_memory.apply,
            x=batch["img"].numpy().reshape(-1),
            y_true=batch["regression"].numpy(),
            opt_state=opt_state,
            params=params,
            state=state,
        )
        print(f"{batch_idx=}, {loss=}, {state=}")


@pytest.mark.parametrize(
    ("ds_conf", "layer_feats", "lr"),
    ((((12, 8), 100), [128, 32, 4, 1], 1e-4), (((16, 32), 10000), [128, 128, 64, 64, 32, 1], 1e-4)),
)
def test_supervised_regression_with_mlp(ds, mlp, tx) -> None:

    @jax.jit
    def mse(params, x, y_true):
        def se(x, y_true):
            return (mlp.apply(params, x) - y_true) ** 2 / len(x)

        return jnp.squeeze(jnp.mean(jax.vmap(se)(x, y_true), axis=0))

    grad_mse = jax.value_and_grad(mse)

    params = None
    num_epochs = 50

    first_sample_sum = None

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(ds)
        for batch_idx, batch in enumerate(pbar):

            imgs, y_true = batch["img"].numpy(), batch["regression"].numpy()
            x = imgs.reshape([imgs.shape[0], imgs.size // imgs.shape[0]])

            if batch_idx == 0:
                if first_sample_sum is None:
                    first_sample_sum = x.sum()
                else:
                    assert first_sample_sum == x.sum()

            if not params:
                params = mlp.init(jax.random.key(0), x[0])
                opt_state = tx.init(params)

            loss, grads = grad_mse(params, x, y_true)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            pbar.set_description(f"loss={float(loss):.2f} {epoch}/{num_epochs}")

        y = mlp.apply(params, x[0])
        print(f"{y_true[0]=}, {y=}")


@pytest.mark.parametrize(
    ("mog_conf", "layer_feats", "lr", "epochs"),
    (
        (
            {"samples": 1600, "locs": [[0, 0, 0], [10, 10, 10], [5, 5, 5]], "stddevs": [1, 1, 1]},
            [32, 32, 32, 3],
            1e-3,
            4,
        ),
        (
            {"samples": 16000, "locs": [[0, 0, 0], [10, 10, 10], [5, 5, 5]], "stddevs": [2, 2, 2]},
            [64, 64, 64, 64, 3],
            1e-3,
            20,
        ),
    ),
)
def test_supervised_mog_classification_mlp(mog_ds, mlp, tx, epochs) -> None:

    @jax.jit
    def mce(params, x, y_true):
        def ce(x, y_true):
            logits = mlp.apply(params, x)
            probs = jax.nn.log_softmax(logits)
            one_hot_labels = jax.nn.one_hot(y_true, logits.shape[-1])
            return -jnp.sum(one_hot_labels * probs, axis=-1)

        return jnp.squeeze(jnp.mean(jax.vmap(ce)(x, y_true), axis=0))

    grad_mce = jax.value_and_grad(mce)

    params = None

    for epoch in range(1, epochs + 1):
        pbar = tqdm(zip(*mog_ds))
        for feats, targets in pbar:

            if not params:
                params = mlp.init(jax.random.key(0), feats[0])
                opt_state = tx.init(params)

            loss, grads = grad_mce(params, feats, targets)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            pbar.set_description(f"loss={float(loss):.2f} {epoch}/{epochs}")

        y_true = mog_ds[1].reshape(-1)
        y_logits = mlp.apply(params, mog_ds[0].reshape((1, -1, mog_ds[0].shape[-1])))
        y_pred = y_logits.argmax(axis=-1).reshape(-1)
    
        accuracy = float((y_true == y_pred).mean())
        print(f"{accuracy=}\n\n")

    assert accuracy > 0.98