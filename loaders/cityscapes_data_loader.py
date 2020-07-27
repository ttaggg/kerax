"""Data loader for Cityscapes segmentation dataset.

Link: https://www.cityscapes-dataset.com/
"""

import collections
import glob
import os

import numpy as np
from absl import logging

from generators import cityscapes_generator
from loaders import data_loader


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

        images_dir = os.path.join(self._config['data_path'],
                                  'leftImg8bit_trainvaltest')
        masks_dir = os.path.join(self._config['data_path'],
                                 'gtFine_trainvaltest')
        # We are going to use part of train dataset for validation.
        all_images = glob.glob(f'{images_dir}/leftImg8bit/train/*/*')
        all_masks = glob.glob(f'{masks_dir}/gtFine/train/*/*labelIds.png')

        datadict = collections.defaultdict(list)
        for image_name in all_images:
            # Cut off _leftImg8bit.png.
            prefix = os.path.basename(image_name)
            prefix = '_'.join(prefix.split('_')[:-1])
            datadict[prefix].append(image_name)
        for mask_name in all_masks:
            # Cut off _gtFine_labelIds.png.
            prefix = os.path.basename(mask_name)
            prefix = '_'.join(prefix.split('_')[:-2])
            datadict[prefix].append(mask_name)

        # Only use samples that have both image and mask.
        pairs = [(v[0], v[1]) for k, v in datadict.items() if len(v) == 2]
        self._images_paths, self._masks_paths = zip(*pairs)
        self._images_paths = np.array(self._images_paths)
        self._masks_paths = np.array(self._masks_paths)

        self._num_folds = self._config.get('n_folds', 20)
        random_state = self._config.get('random_state', 42)
        shuffle = self._config.get('shuffle', True)

        self._folds = self._get_folds(self._images_paths, self._masks_paths,
                                      self._num_folds, random_state, shuffle)
        logging.info('Data paths are loaded.')

    def get_fold(self, fold):
        """Return fold-th fold.

        Args:
            fold: Integer, number of fold to return.
        Returns:
            train_set: List, paths to images for train set.
            test_set: List, paths to images for test set.
        Raises:
            ValueError is fold >= self._num_folds
        """
        if fold >= self._num_folds:
            raise ValueError(f'Fold can be between 0 and {self._num_folds - 1},'
                             f' given: {fold}.')

        logging.info(f'Returning train and test data for the fold: {fold}.')

        train_inx, test_inx = self._folds[fold]

        train_set = list(
            zip(self._images_paths[train_inx], self._masks_paths[train_inx]))
        test_set = list(
            zip(self._images_paths[test_inx], self._masks_paths[test_inx]))

        return train_set, test_set

    def generators(self, batch_size=1):
        """Create and return train and test generators.

        Args:
            batch_size: Integer, batch sizee.
                All other parameters are infered from data_config.
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

        train_generator = cityscapes_generator.CityscapesGenerator(
            dataset=train_set,
            batch_size=batch_size,
            augmentations=train_augmentation,
            is_training=True)
        test_generator = cityscapes_generator.CityscapesGenerator(
            dataset=test_set,
            batch_size=batch_size,
            augmentations=test_augmentation,
            is_training=False)

        return train_generator, test_generator
