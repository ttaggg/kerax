"""Tests for Cityscapes generator."""

import collections

import numpy as np
from absl.testing import absltest

from kerax.loaders import cityscapes_data_loader as cdl


class CityscapesGeneratorTest(absltest.TestCase):

    def _get_loader(self, config):
        loader = cdl.CityscapesDataLoader(config)
        return loader

    def test_getitem(self):
        """Test for __getitem__()."""

        config = collections.defaultdict(collections.defaultdict)
        config.update({
            'data_path':
                'kerax/testdata/cityscapes/images/leftImg8bit_trainvaltest/leftImg8bit',
            'labels_path':
                'kerax/testdata/cityscapes/images/gtFine_trainvaltest/gtFine',
            'task':
                'cityscapes_segmentation',
            'n_folds':
                10,
        })
        loader = self._get_loader(config)

        batch_size = 2
        train_generator, _ = loader.generators(batch_size=batch_size)
        for sample in train_generator:
            x_data, y_data = sample
            image_batch, image_shape = x_data.shape[0], x_data.shape[1:]
            mask_batch, mask_shape = y_data.shape[0], y_data.shape[1:]
            self.assertLessEqual(image_batch, 2)
            self.assertEqual(image_batch, mask_batch)
            self.assertEqual(image_shape, (1024, 2048, 3))
            self.assertEqual(mask_shape, (1024, 2048, 34))

    def test_getitem_ignore_labels(self):
        """Test for __getitem__() with ignore_labels."""

        config = collections.defaultdict(collections.defaultdict)
        config.update({
            'data_path':
                'kerax/testdata/cityscapes/images/leftImg8bit_trainvaltest/leftImg8bit',
            'labels_path':
                'kerax/testdata/cityscapes/images/gtFine_trainvaltest/gtFine',
            'task':
                'cityscapes_segmentation',
            'n_folds':
                10,
            'ignore_labels': [1, 2, 3, 4, 5]
        })
        loader = self._get_loader(config)

        batch_size = 2
        train_generator, _ = loader.generators(batch_size=batch_size)
        for sample in train_generator:
            x_data, y_data = sample
            image_batch, image_shape = x_data.shape[0], x_data.shape[1:]
            mask_batch, mask_shape = y_data.shape[0], y_data.shape[1:]
            self.assertLessEqual(image_batch, 2)
            self.assertEqual(image_batch, mask_batch)
            self.assertEqual(image_shape, (1024, 2048, 3))
            self.assertEqual(mask_shape, (1024, 2048, 34 - 5))
            self.assertLessEqual(np.amax(y_data), 34 - 5)


if __name__ == '__main__':
    absltest.main()
