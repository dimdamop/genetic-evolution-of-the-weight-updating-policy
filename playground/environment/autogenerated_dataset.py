from functools import partial
from pathlib import Path
from math import floor
from random import randrange
import numpy as np
import pandas as pd
from PIL import Image


MIN_IMG_LEN = 80
MAX_IMG_LEN = 120
NUM_LAYERS = 8
GT_LAYER_IDX = 5


def ellipse(
    img: np.array,
    hlen: float | int = 0.4,
    vlen: float | int | None = None,
    fully_within_image: bool = True,
    ellipse_intensity: int | str = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adds a randomly sized ellipse to the provided image, at a random position."""

    intensity_calc = {
        "mean": np.mean,
        "median": np.median,
        "mean-per-channel": partial(np.mean, axis=(0, 1)),
        "median-per-channel": partial(np.median, axis=(0, 1)),
    }

    if isinstance(ellipse_intensity, str):
        ellipse_intensity = intensity_calc(img)

    if vlen is None:
        vlen = hlen

    img_vlen, img_hlen = img.shape[:2]
    img_size = min(img_hlen, img_vlen)

    if not isinstance(hlen, int):
        hlen = floor(img_size * hlen)

    if not isinstance(vlen, int):
        vlen = floor(img_size * vlen)

    hrad = hlen // 2
    vrad = vlen // 2

    if hrad <= 0 or vrad <= 0:
        return img

    if fully_within_image:
        hrad = min(hrad, img_hlen // 2)
        vrad = min(vrad, img_hlen // 2)

        ch_start = hrad
        cv_start = vrad
    else:
        ch_start, cv_start = 0, 0

    ctr_h = randrange(ch_start, img_hlen - ch_start)
    ctr_v = randrange(cv_start, img_hlen - cv_start)

    yy, xx = np.mgrid[0:img_vlen, 0:img_hlen]
    ellipse_mask = ((xx - ctr_h) / hrad) ** 2 + ((yy - ctr_v) / vrad) ** 2 <= 1

    if isinstance(ellipse_intensity, np.ndarray):
        img[ellipse_mask, :] = ellipse_intensity
    else:
        img[ellipse_mask] = ellipse_intensity

    return img, ellipse_mask


def sample_element():
    height = np.random.randint(MIN_IMG_LEN, MAX_IMG_LEN)
    width = np.random.randint(MIN_IMG_LEN, MAX_IMG_LEN)
    img = np.random.randint(255, size=[height, width, 3]).astype(np.int32)

    layer_vals = sorted(np.random.randint(255, size=NUM_LAYERS))
    for layer_idx, layer_val in enumerate(layer_vals):
        img, mask = ellipse(img, hlen=np.random.random() / 2 + 0.1, ellipse_intensity=layer_val)

        if layer_idx == GT_LAYER_IDX:
            retmask = mask

    masked_img_mean = img[retmask].mean()
    binary_tgt = (masked_img_mean < 196).astype(np.int32)
    regression_tgt = (masked_img_mean - img.mean()).astype(np.float32)

    yield img, retmask, binary_tgt, regression_tgt
