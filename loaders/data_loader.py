"""Parent class for dataset loaders."""

import abc
import os

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

    def _get_folds(self, x_data, y_data, num_folds, random_state, shuffle):
        """Split x_data and y_data in train / test folds."""
        kfmeter = model_selection.KFold(
            n_splits=num_folds, random_state=random_state, shuffle=shuffle)
        folds = kfmeter.split(x_data, y_data)
        folds = list(folds)
        return folds

    @abc.abstractmethod
    def get_fold(self, fold):
        """Return fold-th fold.
        Args:
            fold: Integer, number of fold to return.
        Returns:
            train_set: List, paths to images for train set.
            test_set: List, paths to images for test set.
        """

    @abc.abstractmethod
    def generators(self, batch_size=1):
        """Create and return train and test generators.

        Args:
            batch_size: Integer, batch sizee.
                All other parameters are infered from data_config.
        Returns:
            train_generator: Train generator.
            test_generator: Train generator.
        """
