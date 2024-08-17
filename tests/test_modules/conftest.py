from typing import Tuple

import numpy as np
import optax
import pytest

from ..artificial_dataset import batch_iterator
from evolvable_neuron.modules.base import MLP, MLPWithMemory


BATCH_SIZE = 16


@pytest.fixture
def ds(ds_conf: Tuple[Tuple[int, int], int | None]):
    resize_to, take = ds_conf
    ds = batch_iterator(batch_size=BATCH_SIZE, resize_to=resize_to)

    if take is not None:
        ds = [sample for sample in ds.take(take)]

    return ds


@pytest.fixture
def tx(lr: float):
    return optax.adam(learning_rate=lr)


@pytest.fixture
def mlp(layer_feats: Tuple[int]):
    return MLP(layer_feats=layer_feats)


@pytest.fixture
def mlp_with_memory(layer_feats: Tuple[int]):
    return MLPWithMemory(layer_feats=layer_feats)


@pytest.fixture
def mog_ds(mog_conf: dict, batch_size: int | None = 16) -> tuple[np.ndarray]:
    """
    Generates a mixture of sphrerical Gaussians dataset.

    Args:
        mog_conf (dict): Configuration dictionary with the following keys:
            - "samples": Number of samples to generate.
            - "locs": Array of shape (k, d) with the d-dimensional centers of k Gaussians.
            - "stddevs": Array of shape (k,) with the standard deviations of the Gaussians.

    Returns:
        tuple: A tuple containing:
            - features: Array of shape (n, d) with the generated features.
            - labels: Array of shape (n, 1) with the corresponding Gaussian labels.
    """

    locs = np.array(mog_conf["locs"])

    n = mog_conf["samples"]
    k, d = locs.shape

    # Sample Gaussian indices uniformly
    labels = np.random.randint(0, k, size=n)

    # Generate features
    features = np.zeros((n, d))
    for i in range(k):
        indices = np.where(labels == i)[0]
        std = mog_conf["stddevs"][i]
        features[indices, :] = np.random.randn(len(indices), d) * std + locs[i].reshape(1, -1)

    def batch(x):
        if batch_size is None:
            return x

        return np.squeeze(x.reshape((len(x) // batch_size, batch_size, -1)))

    return batch(features), batch(labels)
