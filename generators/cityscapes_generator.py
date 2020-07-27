""""Generators for Cityscapes segmentation dataset."""

import numpy as np
from keras import preprocessing

from generators import generator


def _load_image(path):
    return np.asarray(preprocessing.image.load_img(path)) / 255.


def _load_mask(path):
    return np.asarray(
        preprocessing.image.load_img(path, color_mode='grayscale'))


class CityscapesGenerator(generator.Generator):
    """Custom generator for segmentation.

    Args:
        :same as parent class.
    """

    def __init__(self, dataset, batch_size, augmentations, is_training):
        super().__init__(dataset, batch_size, augmentations, is_training)
        self._num_classes = 35

    def __getitem__(self, idx):
        batch = self._pairs[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch = [self._to_images(el) for el in batch]
        x_data, y_data = tuple(zip(*batch))
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        return x_data, y_data

    def _to_images(self, pair):
        image_path, mask_path = pair
        image = _load_image(image_path)
        mask = _load_mask(mask_path)
        mask = np.eye(self._num_classes)[mask]
        image, mask = self._augment((image, mask))
        return image, mask

    def _augment(self, pairs):
        image, mask = pairs
        data = {'image': image, 'mask': mask}
        aug_data = self._augmentations(**data)
        return aug_data['image'], aug_data['mask']
