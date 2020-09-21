"""Data loader for MAuto speed estimation."""

import glob
import re

import numpy as np
from absl import logging

from kerax.generators import mauto_generator
from kerax.loaders import data_loader


class MAutoDataLoader(data_loader.DataLoader):
    """Load paths to images and labels.

    Args:
        config: Dictionary with data configs, with
            mandatory "data_path" string.
    Raises:
        FileNotFoundError if no data_path was provided
            or path was not found.
    """

    def __init__(self, config):
        super().__init__(config)
        # Input images should be optical flow frames in RGB encoding.
        # Names are of the type: 'frame_INT.jpg' -> sort arithmetically.
        all_images = glob.glob(f'{config["data_path"]}/*.jpg')
        all_images = sorted(
            all_images, key=lambda x: int(re.split(r'\.|_', x)[-2]))

        labels_path = config['labels_path']
        if labels_path == 'prediction':
            all_labels = [0.] * len(all_images)
        else:
            all_labels = np.loadtxt(labels_path, dtype=float)

        assert len(all_images) == len(all_labels)

        self._image_paths = np.array(all_images)
        self._labels = np.array(all_labels)

        # Set shuffle to False in config to get sequential data
        # for checking how moving average works and for debugging reasons.
        # In this case additionally shuffle train set only before first epoch.
        random_state = config.get('random_state', None)
        shuffle = config.get('shuffle', False)
        self._num_folds = self._config.get('n_folds', 10)
        self._folds = self._split_dataset(
            self._image_paths,
            self._labels,
            self._num_folds,
            random_state=random_state,
            shuffle=shuffle)
        logging.info('Data paths and labels are loaded.')

    def generators(self, batch_size=1):
        """Create and return train and test generators.

        Args:
            batch_size: Integer, batch size.
                All other parameters are inferred from data_config.
        Returns:
            train_generator: Train generator.
            test_generator: Train generator.
        """

        fold = self._config.get('fold', 0)
        train_set, test_set = self.get_fold(fold)

        train_augmentation = self._config['augmentation'].get('train', {})
        test_augmentation = self._config['augmentation'].get('test', {})

        np.random.shuffle(train_set)
        train_generator = mauto_generator.MAutoGenerator(
            dataset=train_set,
            batch_size=batch_size,
            augmentations=train_augmentation,
            is_training=True)
        test_generator = mauto_generator.MAutoGenerator(
            dataset=test_set,
            batch_size=batch_size,
            augmentations=test_augmentation,
            is_training=False)

        return train_generator, test_generator

    def prediction(self, batch_size=1):

        test_augmentation = self._config['augmentation'].get('test', {})
        test_generator = mauto_generator.MAutoGenerator(
            dataset=list(zip(self._image_paths, self._labels)),
            batch_size=batch_size,
            augmentations=test_augmentation,
            is_training=False)

        return test_generator
