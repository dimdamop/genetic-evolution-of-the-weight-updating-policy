from typing import Tuple

import flax
import jax
import optax
import pytest


@pytest.mark.parametrize(
    ("resize_to", "layer_feats", "lr", "with_memory"),
    (((18, 28), [128, 32, 1], 1e-3, True), ((64, 96), [128, 128, 64, 64, 32, 1], 1e-4), True),
)
def test_supervised_regression_with_mlp_with_memory(ds, mlp, tx) -> None:

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
    variables = mlp_module.init(jax.random.key(0), x)
    state, params = flax.core.pop(variables, "params")
    del variables
    opt_state = tx.init(params)

    for batch_idx, batch in enumerate(ds):
        opt_state, params, state, loss = update_step(
            apply_fn=mlp_module.apply,
            x=batch["img"].numpy().reshape(-1),
            y_true=batch["regression"].numpy(),
            opt_state=opt_state,
            params=params,
            state=state,
        )
        print(f"{batch_idx=}, {loss=}, {state=}")

        if batch_idx == 99:
            break
