"""Image-related utils."""

import io
import random

import numpy as np
import tensorflow as tf
from PIL import Image


def image_to_summary(np_image):
    """Make TF summary.

    Args:
        np_image: Numpy array, image.
    Returns:
        TF summary.
    """
    height, width, channel = np_image.shape
    image = Image.fromarray((255. * np_image).astype(np.uint8))
    output = io.BytesIO()
    image.convert('RGB').save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(
        height=height,
        width=width,
        colorspace=channel,
        encoded_image_string=image_string)


def mask_to_rle(mask):
    """Make run-length encoded representation from image.
    Reference:
        https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

    Args:
        mask: Numpy array, mask. Background is 0.
    Returns:
        mask_rle: String: Run-length encoded representation.
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(mask_rle, shape):
    """Make image mask from run-length encoded representation.

    Args:
        mask_rle: String: Run-length encoded representation.
        shape: list/tuple: shape of the mask.
    Returns:
        mask: Numpy array, mask. Background is 0.
    """
    mask = []
    rle_list = [int(x) for x in mask_rle.split()]
    prev_inx = 0
    for start, length in zip(rle_list[::2], rle_list[1::2]):
        mask.extend([0.] * (start - prev_inx - 1))
        mask.extend([1.] * length)
        prev_inx = start + length - 1

    num_pixels = shape[0] * shape[1]
    mask.extend([0.] * (num_pixels - len(mask)))

    return np.array(mask).reshape(shape[::-1]).T


def _colors(num):
    palette = []
    red = int(random.random() * 256)
    green = int(random.random() * 256)
    blue = int(random.random() * 256)
    step = 256 / num
    for _ in range(num):
        red += step
        green += step
        blue += step
        red = int(red) % 256
        green = int(green) % 256
        blue = int(blue) % 256
        palette.append([red, green, blue])
    return palette


def create_label_colormap(num_colors):
    """Labels to colors."""
    colormap = np.zeros((num_colors, 3), dtype=int)
    palette = _colors(num_colors)
    for i in range(num_colors):
        colormap[i, ...] = palette[i]
    return colormap
