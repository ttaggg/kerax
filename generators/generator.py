"""Parent class for generators."""

import abc
import albumentations
import numpy as np
from absl import logging
from keras import utils


class Generator(utils.Sequence, abc.ABC):
    """Custom generator class.

    Args:
        dataset: List of sample pairs (image, mask) or (image, labels).
        batch_size: Integer, batch size.
        augmentations: dictionary with "key": "value" being
            "string, name of augmentation": "dict with arguments".
        is_training: Bool, whether this generator is for training or not.
    """

    def __init__(self, dataset, batch_size, augmentations, is_training):
        self._pairs = dataset
        self._size = len(self._pairs)
        self._batch_size = batch_size
        albumentations_dict = {
            'Resize': albumentations.Resize,
            'RandomCrop': albumentations.RandomCrop,
            'HorizontalFlip': albumentations.HorizontalFlip,
            'VerticalFlip': albumentations.VerticalFlip,
            'RandomBrightness': albumentations.RandomBrightness,
            'RandomContrast': albumentations.RandomContrast,
            'ShiftScaleRotate': albumentations.ShiftScaleRotate,
        }
        self._augmentations = albumentations.Compose([
            albumentations_dict[augm_name](**augmentations[augm_name])
            for augm_name in augmentations
            if augm_name in albumentations_dict
        ])
        self._is_training = is_training
        logging.info(f'Dataset size: {self._size}; batch size {batch_size}, '
                     f'training: {is_training}.')

    def on_epoch_end(self):
        if self._is_training:
            np.random.shuffle(self._pairs)

    def __len__(self):
        return int(np.ceil(self._size / float(self._batch_size)))

    def _augment(self, pairs):
        """Apply augmentations to the dataset.

        Override this optional method to apply augmentations
        to images or images and masks.
        """
        return pairs

    @property
    def batch_size(self):
        """Return batch size."""
        return self._batch_size

    @abc.abstractmethod
    def __getitem__(self, idx):
        """__getitem__ method of keras.utils.Sequence"""
