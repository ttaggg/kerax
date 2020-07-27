"""Tests for general_utils.py"""

import collections
import os
import shutil
import tempfile
from absl.testing import absltest

from utils import general_utils as gutils


class GeneralUtilsTest(absltest.TestCase):

    def test_get_config(self):
        """Test for get_config()."""

        # Check whether minimal config is parsed without problems.
        gutils.get_config('testdata/minimal_config.py')

        config = gutils.get_config('testdata/maximal_config.py')

        def _recursive_check(config):
            if isinstance(config, dict):
                if not isinstance(config, collections.defaultdict):
                    raise AssertionError
                for key in config:
                    _recursive_check(config[key])

        # Check whether all dicts in config are defaultdicts.
        _recursive_check(config)

        # Raise error if config does not exist.
        with self.assertRaises(FileNotFoundError):
            gutils.get_config('testdata/nonexisting_config.py')

        # Small sanity check for required fields.
        gutils.check_required_config_fields(config, only_predict=False)
        config['model'].pop('input_shape')
        with self.assertRaises(AssertionError):
            gutils.check_required_config_fields(config, only_predict=False)
        # We do not need model config in prediction mode.
        gutils.check_required_config_fields(config, only_predict=True)
        # We still need data config even in prediction mode.
        config['data'].pop('data_path')
        with self.assertRaises(AssertionError):
            gutils.check_required_config_fields(config, only_predict=True)

    def test_get_model(self):
        """Test for get_model()."""

        ckpt = 'testdata/chpt.01.hdf5'
        ckpt_dir = tempfile.TemporaryDirectory()
        shutil.copyfile(ckpt, os.path.join(ckpt_dir.name, 'chpt.01.hdf5'))
        shutil.copyfile(ckpt, os.path.join(ckpt_dir.name, 'chpt.40.hdf5'))
        shutil.copyfile(ckpt, os.path.join(ckpt_dir.name, 'chpt.100.hdf5'))

        # Raise if no such file.
        with self.assertRaises(FileNotFoundError):
            gutils.get_model('testdata/empty_dir')

        # Raise error if empty directory.
        with self.assertRaises(FileNotFoundError):
            gutils.get_model(tempfile.TemporaryDirectory().name)

        # Check whether epoch is correct if directory is given.
        _, epoch = gutils.get_model(ckpt_dir.name)
        self.assertEqual(epoch, 100)

        # Check whether epoch is correct if path to file is given.
        _, epoch = gutils.get_model(os.path.join(ckpt_dir.name, 'chpt.01.hdf5'))
        self.assertEqual(epoch, 1)


if __name__ == '__main__':
    absltest.main()
