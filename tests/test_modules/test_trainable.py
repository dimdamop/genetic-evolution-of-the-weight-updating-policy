from typing import Tuple

import flax
import jax
import optax
import pytest

from jax import numpy as jnp
from tqdm import tqdm


@pytest.mark.parametrize(
    ("resize_to", "layer_feats", "lr"),
    (((18, 28), [128, 32, 1], 1e-3), ((64, 96), [128, 128, 64, 64, 32, 1], 1e-4)),
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

        if batch_idx == 99:
            break


@pytest.mark.parametrize(
    ("resize_to", "take", "layer_feats", "lr"),
    (((8, 12), 200, [128, 32, 1], 1e-3), ((16, 32), 200, [128, 128, 64, 64, 32, 1], 1e-4)),
)
def test_supervised_regression_with_mlp(ds, mlp, tx) -> None:

    @jax.jit
    def mse(params, x, y_true):
        def se(x, y_true):
            return (mlp.apply(params, x) - y_true) ** 2

        return jnp.squeeze(jnp.mean(jax.vmap(se)(x, y_true), axis=0))

    grad_mse = jax.value_and_grad(mse)

    params = None
    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(ds)
        for batch_idx, batch in enumerate(pbar):

            imgs, y_true = batch["img"].numpy(), batch["regression"].numpy()
            x = imgs.reshape([imgs.shape[0], imgs.size // imgs.shape[0]])

            if not params:
                params = mlp.init(jax.random.key(0), x[0])
                opt_state = tx.init(params)

            loss, grads = grad_mse(params, x, y_true)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            pbar.set_description(f"loss={float(loss):.2f} {epoch}/{num_epochs}")
