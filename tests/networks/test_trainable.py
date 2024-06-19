from typing import Tuple

import flax
import jax
import optax
import pytest


@pytest.mark.parametrize(
    ("resize_to", "layer_feats", "lr"),
    (((16, 32), [32, 32, 1], 1e-3), ((48, 32), [32, 32, 8, 1], 1e-4)),
)
def test_batched_supervised_regression(ds, network, tx) -> None:

    def update_step(apply_fn, batch, opt_state, params, state):
        def loss(params):
            y_pred, updated_state = apply_fn(
                {"params": params, **state}, x, mutable=list(state.keys())
            )
            l = ((y_pred - y_true) ** 2).sum()
            return l, updated_state

        batched_loss = lambda x, y: loss()
        (l, updated_state), grads = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return opt_state, params, updated_state, l

    batch = next(iter(ds))
    img = batch["img"][0].numpy().reshape(-1)
    variables = network.init(jax.random.key(0), img)
    state, params = flax.core.pop(variables, "params")
    del variables
    opt_state = tx.init(params)

    batched_update_step = jax.vmap(
        lambda x, y: update_step(
            apply_fn=network.apply,
            x=x.reshape(-1),
            y_true=y,
            opt_state=opt_state,
            params=params,
            state=state,
        )
    )

    for batch_idx, batch in enumerate(ds):
        opt_state, params, state, loss = batched_update_step(
            x=batch["img"].numpy(), y=batch["regression"].numpy(),
        )
        print(loss)