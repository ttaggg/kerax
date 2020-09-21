"""Parent class for dataset loaders."""

import abc
import os

from absl import logging
from sklearn import model_selection


class DataLoader(abc.ABC):
    """Load paths to images.

    Args:
        config: Dictionary with data configs, with
            mandatory "data_path" string.
    """

    def __init__(self, config):
        if not os.path.exists(config['data_path']):
            raise FileNotFoundError(f'{config["data_path"]} was not found.')
        self._config = config
        self._num_folds = None
        self._folds = None
        self._image_paths = None
        self._labels = None

    def _split_dataset(self, x_data, y_data, num_folds, random_state, shuffle):
        """Split x_data and y_data in train / test folds."""
        kfmeter = model_selection.KFold(
            n_splits=num_folds, random_state=random_state, shuffle=shuffle)
        folds = kfmeter.split(x_data, y_data)
        folds = list(folds)
        return folds

    def get_fold(self, fold):
        """Return fold-th fold.

        Args:
            fold: Integer, number of fold to return.
        Returns:
            train_set: List, paths to images for train set.
            test_set: List, paths to images for test set.
        Raises:
            ValueError if fold >= self._num_folds
        """
        if fold >= self._num_folds:
            raise ValueError(f'Fold can be between 0 and {self._num_folds - 1},'
                             f' given: {fold}.')

        logging.info(f'Returning train and test data for the fold: {fold}.')

        train_inx, test_inx = self._folds[fold]
        train_set = list(
            zip(self._image_paths[train_inx], self._labels[train_inx]))
        test_set = list(
            zip(self._image_paths[test_inx], self._labels[test_inx]))
        return train_set, test_set

    @abc.abstractmethod
    def generators(self, batch_size=1):
        """Create and return train and test generators.

        Args:
            batch_size: Integer, batch size.
                All other parameters are inferred from data_config.
        Returns:
            train_generator: Train generator.
            test_generator: Train generator.
        """

    @abc.abstractmethod
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
