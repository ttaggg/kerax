"""Data loader for Cityscapes segmentation dataset.

Link: https://www.cityscapes-dataset.com/
"""

import collections
import glob
import os

import numpy as np
from absl import logging

from kerax.generators import cityscapes_generator
from kerax.loaders import data_loader


class CityscapesDataLoader(data_loader.DataLoader):
    """Load paths to images masks.

    Args:
        config: Dictionary with data configs, with
            mandatory "data_path" string.
    Raises:
        FileNotFoundError if no data_path was provided
            or path was not found.
    """

    def __init__(self, config):
        super().__init__(config)

        # We are going to use part of train dataset for validation.
        images_dir = self._config['data_path']
        all_images = glob.glob(f'{images_dir}/train/*/*')

        masks_dir = self._config['labels_path']
        all_masks = glob.glob(f'{masks_dir}/train/*/*labelIds.png')

        datadict = collections.defaultdict(list)
        for image_name in all_images:
            # Cut off _leftImg8bit.png.
            prefix = os.path.basename(image_name)
            prefix = '_'.join(prefix.split('_')[:-1])
            datadict[prefix].append(image_name)
        for mask_name in all_masks:
            # Cut off _gtFine_labelIds.png or _gtCoarse_labelIds.png.
            prefix = os.path.basename(mask_name)
            prefix = '_'.join(prefix.split('_')[:-2])
            datadict[prefix].append(mask_name)

        # Only use samples that have both image and mask.
        pairs = [(v[0], v[1]) for k, v in datadict.items() if len(v) == 2]
        self._image_paths, self._labels = zip(*pairs)
        self._image_paths = np.array(self._image_paths)
        self._labels = np.array(self._labels)

        self._num_folds = self._config.get('n_folds', 20)
        random_state = self._config.get('random_state', 42)
        shuffle = self._config.get('shuffle', True)

        self._folds = self._split_dataset(self._image_paths, self._labels,
                                          self._num_folds, random_state,
                                          shuffle)
        logging.info('Data paths are loaded.')

    def generators(self, batch_size=1):
        """Create and return train and test generators.

        Args:
            batch_size: Integer, batch size.
                All other parameters are inferred from data_config.
        Returns:
            train_generator: Train generator.
            test_generator: Train generator.
        Raises:
            ValueError if task is not supported for this data.
        """

        fold = self._config.get('fold', 0)
        train_set, test_set = self.get_fold(fold)

        train_augmentation = self._config['augmentation'].get('train', {})
        test_augmentation = self._config['augmentation'].get('test', {})

        ignore_labels = self._config.get('ignore_labels', [])
        kwargs = {'ignore_labels': ignore_labels}

        train_generator = cityscapes_generator.CityscapesGenerator(
            dataset=train_set,
            batch_size=batch_size,
            augmentations=train_augmentation,
            is_training=True,
            **kwargs)
        test_generator = cityscapes_generator.CityscapesGenerator(
            dataset=test_set,
            batch_size=batch_size,
            augmentations=test_augmentation,
            is_training=False,
            **kwargs)

        return train_generator, test_generator

    def prediction(self, batch_size=1):
        """Return the generator for all files that were read
            without train / val split.

        Args:
            batch_size: Integer, batch size.
                All other parameters are inferred from data_config.
        Returns:
            generator: Generator with all images and labels,
                test_augmentations are applied.
        """
        test_augmentation = self._config['augmentation'].get('test', {})
        ignore_labels = self._config.get('ignore_labels', [])
        kwargs = {'ignore_labels': ignore_labels}

        return cityscapes_generator.CityscapesGenerator(
            dataset=list(zip(self._image_paths, self._labels)),
            batch_size=batch_size,
            augmentations=test_augmentation,
            is_training=False,
            **kwargs)
