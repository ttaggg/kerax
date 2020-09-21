"""Data loader for Severstal defect detection dataset."""

import collections
import os

import numpy as np
from absl import logging

from kerax.generators import severstal_generator
from kerax.loaders import data_loader


class SeverstalDataLoader(data_loader.DataLoader):
    """Load paths to images and RLE masks.

    Args:
        config: Dictionary with data configs, with
            mandatory "data_path" and "labels_path" string.
    Raises:
        FileNotFoundError if no data_path was provided
            or path was not found.
    """

    def __init__(self, config):
        super().__init__(config)

        data_dir = self._config['data_path']
        data = np.genfromtxt(
            self._config['labels_path'],
            delimiter=',',
            skip_header=1,
            dtype=None,
            encoding=None)

        df_dict = collections.defaultdict(list)
        for imageid_classid, pixels in data:
            classid = int(imageid_classid[-1])
            imageid = imageid_classid[:-2]
            image_path = os.path.join(data_dir, imageid)
            df_dict[image_path].append((classid, pixels))

        self._num_folds = self._config.get('n_folds', 20)
        random_state = self._config.get('random_state', 42)
        shuffle = self._config.get('shuffle', True)

        self._image_paths, self._labels = zip(*df_dict.items())
        self._folds = self._split_dataset(self._image_paths, self._labels,
                                          self._num_folds, random_state,
                                          shuffle)
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

        Override parent function, because:
            If we directly use numpy indices altogether
            to make a fancy use of broadcasting, we will get OOM,
            because for each string it assigns the space that is enough
            for the longest one in the array, which is too much for RLE masks.
        """
        if fold >= self._num_folds:
            raise ValueError(f'Fold can be between 0 and {self._num_folds - 1},'
                             f' given: {fold}.')

        logging.info(f'Returning train and test data for the fold: {fold}.')

        train_inx, test_inx = self._folds[fold]
        train_set = []
        for inx in train_inx:
            train_set.append((self._image_paths[inx], self._labels[inx]))
        test_set = []
        for inx in test_inx:
            test_set.append((self._image_paths[inx], self._labels[inx]))

        return train_set, test_set

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

        if self._config['task'].endswith('segmentation'):
            generator = severstal_generator.SegmentationGenerator
            include_background = self._config.get('include_background', True)
            kwargs = {'include_background': include_background}
        elif self._config['task'].endswith('classification'):
            generator = severstal_generator.ClassificationGenerator
            num_channels = self._config.get('num_channels', 1)
            kwargs = {'num_channels': num_channels}
        else:
            raise ValueError(
                'Supported tasks are "severstal_segmentation" and'
                f'"severstal_classification", given: {self._config["task"]}.')

        train_augmentation = self._config['augmentation'].get('train', {})
        test_augmentation = self._config['augmentation'].get('test', {})

        train_generator = generator(
            dataset=train_set,
            batch_size=batch_size,
            augmentations=train_augmentation,
            is_training=True,
            **kwargs)
        test_generator = generator(
            dataset=test_set,
            batch_size=batch_size,
            augmentations=test_augmentation,
            is_training=False,
            **kwargs)

        return train_generator, test_generator

    def prediction(self, batch_size=1):
        raise NotImplementedError
