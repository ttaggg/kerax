"""Tests for MAuto data loader.py"""

import collections

from absl.testing import absltest

from kerax.loaders import mauto_data_loader as mdl


class MAutoDataLoaderTest(absltest.TestCase):

    def test_init(self):
        """Test for __init__()."""
        # pylint: disable= protected-access

        config = collections.defaultdict(collections.defaultdict)
        config.update({
            'data_path': 'kerax/testdata/mauto/images/opticalflow',
            'labels_path': 'kerax/testdata/mauto/train.txt',
            'task': 'mauto_regression',
            'n_folds': 10,
        })
        loader = mdl.MAutoDataLoader(config)

        self.assertEqual(loader._num_folds, 10)
        self.assertLen(loader._image_paths, 19)
        self.assertLen(loader._labels, 19)

    def test_get_fold(self):
        """Test for get_fold()."""

        config = collections.defaultdict(collections.defaultdict)
        config.update({
            'data_path': 'kerax/testdata/mauto/images/opticalflow',
            'labels_path': 'kerax/testdata/mauto/train.txt',
            'task': 'mauto_regression',
            'n_folds': 10,
        })
        loader = mdl.MAutoDataLoader(config)

        for i in range(10):
            for j in range(10):
                train_i, test_i = loader.get_fold(i)
                train_j, test_j = loader.get_fold(j)
                if i == j:
                    continue
                # Check whether they are different.
                self.assertNotEqual(train_i, train_j)
                self.assertNotEqual(test_i, test_j)
                # Check whether there is no leak.
                self.assertNoCommonElements(test_i, train_i)
                self.assertNoCommonElements(test_j, train_j)


if __name__ == '__main__':
    absltest.main()
