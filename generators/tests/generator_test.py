"""Test for Generator class."""

import collections

import numpy as np
from absl.testing import absltest

from loaders import severstal_data_loader as sdl


class GeneratorTest(absltest.TestCase):

    def setUp(self):
        self._config = collections.defaultdict(collections.defaultdict)
        self._config.update({
            'data_path': 'testdata/severstal/test_set.csv',
            'task': 'severstal_segmentation',
            'n_folds': 10,
        })
        self._loader = sdl.SeverstalDataLoader(self._config)

    def test_init(self):
        """Test for __init__()."""
        # pylint: disable= protected-access

        # Check main attributes.
        train_generator, test_generator = self._loader.generators(batch_size=2)
        self.assertLen(train_generator._pairs, 22)
        self.assertEqual(train_generator._batch_size, 2)
        self.assertEqual(train_generator._is_training, True)
        self.assertLen(test_generator._pairs, 3)
        self.assertEqual(test_generator._batch_size, 2)
        self.assertEqual(test_generator._is_training, False)

        # Check whether batches are divided correctly.
        batch_sizes = [1, 2, 3, 4, 5]
        for batch_size in batch_sizes:
            train_generator, test_generator = self._loader.generators(
                batch_size=batch_size)
            self.assertLen(train_generator, np.ceil(22 / batch_size))
            self.assertLen(test_generator, np.ceil(3 / batch_size))

    def test_yield_elements(self):
        """Test for yield_elements()."""
        train_generator, test_generator = self._loader.generators(batch_size=2)

        train_images_epoch_1 = [x for x, _ in train_generator.yield_elements()]
        train_images_epoch_1 = np.concatenate(train_images_epoch_1)

        train_images_epoch_2 = [x for x, _ in train_generator.yield_elements()]
        train_images_epoch_2 = np.concatenate(train_images_epoch_2)

        # Train images should be shuffled after one epoch.
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(train_images_epoch_1,
                                          train_images_epoch_2)

        test_images_epoch_1 = [x for x, _ in test_generator.yield_elements()]
        test_images_epoch_1 = np.concatenate(test_images_epoch_1)

        test_images_epoch_2 = [x for x, _ in test_generator.yield_elements()]
        test_images_epoch_2 = np.concatenate(test_images_epoch_2)

        # Test samples should not be shuffled after one epoch.
        np.testing.assert_array_equal(test_images_epoch_1, test_images_epoch_2)
