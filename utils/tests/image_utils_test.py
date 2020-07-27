"""Tests for image_utils.py"""

import numpy as np
from absl.testing import absltest

from utils import image_utils


class ImageUtilsTest(absltest.TestCase):

    def test_mask_to_rle(self):
        """Test for mask_to_rle()."""

        mask = np.array([
            [0., 0., 1., 1.],
            [0., 1., 1., 0.],
            [1., 0., 1., 1.],
            [1., 1., 0., 0.],
            [0., 1., 0., 0.],
        ])
        # From top to bottom, from left to right.
        # Indices start from 1.
        # 3 2: starting from third pixel we have two defect pixels.
        # 7 1: starting from seventh pixel we have one defect pixel.
        # 9 5: starting from nineth pixel we have five defect pixels.
        # <...>
        expected_rle = '3 2 7 1 9 5 16 1 18 1'
        rle = image_utils.mask_to_rle(mask)
        self.assertEqual(expected_rle, rle)

    def test_rls_to_mask(self):
        """Test for rle_to_mask()."""

        rle = '3 2 7 1 9 5 16 1 18 1'
        expected_mask = np.array([
            [0., 0., 1., 1.],
            [0., 1., 1., 0.],
            [1., 0., 1., 1.],
            [1., 1., 0., 0.],
            [0., 1., 0., 0.],
        ])
        mask = image_utils.rle_to_mask(rle, shape=expected_mask.shape)
        np.testing.assert_array_equal(expected_mask, mask)

    def test_rle_msk_random(self):
        """Test for rle_to_mask() and mask_to_rle()."""

        for _ in range(100):
            mask = np.random.randint(2, size=42)
            mask = mask.reshape((6, 7))

            rle = image_utils.mask_to_rle(mask)
            recreated_mask = image_utils.rle_to_mask(rle, shape=(6, 7))
            np.testing.assert_array_equal(recreated_mask, mask)


if __name__ == '__main__':
    absltest.main()
