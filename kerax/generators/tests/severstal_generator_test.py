"""Tests for Severstal generators."""

import collections

from absl.testing import absltest

from kerax.loaders import severstal_data_loader as sdl


class SegmentationGeneratorTest(absltest.TestCase):

    def _get_loader(self, config):
        loader = sdl.SeverstalDataLoader(config)
        return loader

    def test_getitem(self):
        """Test for __getitem__()."""

        config = collections.defaultdict(collections.defaultdict)
        config.update({
            'data_path': 'kerax/testdata/severstal/images',
            'labels_path': 'kerax/testdata/severstal/test_set.csv',
            'task': 'severstal_segmentation',
            'n_folds': 10,
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
            self.assertEqual(image_shape, (256, 1600, 1))
            self.assertEqual(mask_shape, (256, 1600, 5))

    def test_getitem_nobackground(self):
        """Test for __getitem__() without background."""

        config = collections.defaultdict(collections.defaultdict)
        config.update({
            'data_path': 'kerax/testdata/severstal/images',
            'labels_path': 'kerax/testdata/severstal/test_set.csv',
            'include_background': False,
            'task': 'severstal_segmentation',
            'n_folds': 10,
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
            self.assertEqual(image_shape, (256, 1600, 1))
            self.assertEqual(mask_shape, (256, 1600, 4))


class ClassificationGeneratorTest(absltest.TestCase):

    def setUp(self):
        self._config = collections.defaultdict(collections.defaultdict)
        self._config.update({
            'data_path': 'kerax/testdata/severstal/images',
            'labels_path': 'kerax/testdata/severstal/test_set.csv',
            'task': 'severstal_classification',
            'n_folds': 10,
        })
        self._loader = sdl.SeverstalDataLoader(self._config)

    def test_getitem(self):
        """Test for __getitem__()."""

        batch_size = 2
        train_generator, _ = self._loader.generators(batch_size=batch_size)
        for sample in train_generator:
            x_data, y_data = sample
            image_batch, image_shape = x_data.shape[0], x_data.shape[1:]
            mask_batch, mask_shape = y_data.shape[0], y_data.shape[1:]
            self.assertLessEqual(image_batch, 2)
            self.assertEqual(image_batch, mask_batch)
            self.assertEqual(image_shape, (256, 1600, 1))
            self.assertEqual(mask_shape, (4,))


if __name__ == '__main__':
    absltest.main()
