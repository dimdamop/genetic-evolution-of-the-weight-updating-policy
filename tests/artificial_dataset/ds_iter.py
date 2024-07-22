from functools import partial
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from .samples import sample_element


NUM_PARALLEL_CALLS = AUTOTUNE
NUM_SAMPLES: int | None = None


if NUM_SAMPLES is not None:
    DS = [sample_element() for _ in range(NUM_SAMPLES)]


def gen():

    idx = 0

    while True:

        if NUM_SAMPLES is not None:
            element = DS[idx]
            idx = (idx + 1) % NUM_SAMPLES
        else:
            element = sample_element()

        img, mask, binary_tgt, regression_tgt = element

        yield {"img": img, "mask": mask, "binary": binary_tgt, "regression": regression_tgt}


def resize_element(element: dict, height: int, width: int) -> dict:

    element["img"] = tf.image.resize_with_pad(
        image=element["img"],
        target_height=height,
        target_width=width,
    )

    element["mask"] = tf.image.resize_with_pad(
        image=tf.expand_dims(element["mask"], -1),
        target_height=height,
        target_width=width,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    return element


def one_hot(element: dict) -> dict:
    element["binary"] = tf.one_hot(element["binary"], depth=2)
    return element


def batch_iterator(batch_size: int | None, resize_to: tuple[int] | None) -> Dataset:

    dataset = Dataset.from_generator(
        gen,
        output_signature={
            "img": tf.TensorSpec(shape=[None, None, 3], dtype=tf.int32),
            "mask": tf.TensorSpec(shape=[None, None], dtype=tf.uint8),
            "binary": tf.TensorSpec(shape=[], dtype=tf.int32),
            "regression": tf.TensorSpec(shape=[], dtype=tf.float32),
        },
    )

    if resize_to is not None:
        dataset = dataset.map(
            partial(resize_element, height=resize_to[0], width=resize_to[1]),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset = dataset.map(one_hot, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    return dataset
