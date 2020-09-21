""""Generators for MAuto speed estimation."""

import numpy as np
from keras import preprocessing

from kerax.generators import generator


def _load_image(path):
    return np.asarray(preprocessing.image.load_img(path))


class MAutoGenerator(generator.Generator):
    """Custom generator for regression.

    Args:
        :same as parent class.
    """

    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch = [self._to_images(el) for el in batch]
        x_data, y_data = tuple(zip(*batch))
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data

    def _to_images(self, pairs):
        image_path, label = pairs

        image = _load_image(image_path)
        # For RGB optical flow only Resize, RandomCrop and Normalize should
        # be used unless we calculate optical flow on the fly from images.
        image = self._augment((image, label))
        return (image, label)

    def _augment(self, pairs):
        image, _ = pairs
        aug_data = self._augmentations(image=image)
        return aug_data['image']
