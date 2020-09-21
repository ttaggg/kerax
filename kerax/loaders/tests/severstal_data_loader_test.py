"""Tests for Severstal data loader.py"""

import collections

import numpy as np
from absl.testing import absltest

from kerax.loaders import severstal_data_loader as sdl


class SeverstalDataLoaderTest(absltest.TestCase):

    def setUp(self):
        self._config = collections.defaultdict(collections.defaultdict)
        self._config.update({
            'data_path': 'kerax/testdata/severstal/images',
            'labels_path': 'kerax/testdata/severstal/test_set.csv',
            'task': 'severstal_segmentation',
            'n_folds': 10,
        })
        self._loader = sdl.SeverstalDataLoader(self._config)

    def test_init(self):
        """Test for __init__()."""
        # pylint: disable= protected-access

        self.assertEqual(self._loader._num_folds, 10)
        self.assertLen(self._loader._image_paths, 25)
        self.assertLen(self._loader._labels, 25)

        masks = self._loader._labels[0]
        self.assertLen(masks, 4)

        defect_type, mask = masks[0]
        self.assertIsInstance(defect_type, int)
        self.assertIsInstance(mask, np.str)

        self.assertLen(self._loader._folds, self._loader._num_folds)

    def test_get_fold(self):
        """Test for get_fold()."""

        # Raises ValueError if fold >= _num_folds.
        with self.assertRaises(ValueError):
            self._loader.get_fold(10)

        # Get two different folds.
        train_0, test_0 = self._loader.get_fold(0)
        train_2, test_2 = self._loader.get_fold(2)

        # Check whether they are different.
        self.assertNotEqual(train_0, train_2)
        self.assertNotEqual(test_0, test_2)

        # Check that split is close to 9:1.
        self.assertLen(train_0, 22)
        self.assertLen(test_0, 3)
        self.assertLen(train_2, 22)
        self.assertLen(test_2, 3)

        # Check train and test sets are not overlapping.
        train_0_images = [image_path for image_path, _ in train_0]
        test_0_images = [image_path for image_path, _ in test_0]
        self.assertNoCommonElements(train_0_images, test_0_images)

        train_2_images = [image_path for image_path, _ in train_2]
        test_2_images = [image_path for image_path, _ in test_2]
        self.assertNoCommonElements(train_2_images, test_2_images)

    def test_generators(self):
        """Test for generators()."""
        # Check whether batches are divided properly.
        batch_sizes = [1, 2, 3, 4, 5]
        for batch_size in batch_sizes:
            train_generator, test_generator = self._loader.generators(
                batch_size=batch_size)
            self.assertLen(train_generator, np.ceil(22 / batch_size))
            self.assertLen(test_generator, np.ceil(3 / batch_size))


if __name__ == '__main__':
    absltest.main()
