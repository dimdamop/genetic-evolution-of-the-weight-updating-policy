from typing import Tuple

import optax
import pytest

from ..artificial_dataset import batch_iterator
from evolvable_neuron.modules.base import MLP


BATCH_SIZE = 16


@pytest.fixture
def ds(resize_to: Tuple[int, int], take: int | None):
    ds = batch_iterator(batch_size=BATCH_SIZE, resize_to=resize_to)

    if take is not None:
        ds = ds.take(take)

    return ds


@pytest.fixture
def tx(lr: float):
    return optax.adam(learning_rate=lr)


@pytest.fixture
def mlp(layer_feats: Tuple[int]):
    return MLP(layer_feats=layer_feats)


@pytest.fixture
def mlp_with_memory(layer_feats: Tuple[int]):
    return MLP(layer_feats=layer_feats)
