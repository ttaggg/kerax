"""Tests for losses.py"""

import numpy as np
import tensorflow as tf

from utils import losses


class LossesTest(tf.test.TestCase):

    def setUp(self):
        np.random.seed(42)
        tf.compat.v1.set_random_seed(42)

    def test_binary_cross_entropy(self):
        """Test for binary_cross_entropy()."""

        y_true = tf.convert_to_tensor([
            [1., 0., 1., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
        ])
        y_pred_corr = tf.convert_to_tensor([
            [1., 0., 1., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
        ])
        y_pred_half_wrong = tf.convert_to_tensor([
            [1., 0., 0., 1.],
            [1., 1., 0., 1.],
            [1., 1., 0., 1.],
            [1., 1., 0., 1.],
        ])
        y_pred_all_wrong = tf.convert_to_tensor([
            [0., 1., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
        ])

        loss_corr = losses.binary_cross_entropy(y_true, y_pred_corr)
        loss_half_wrong = losses.binary_cross_entropy(y_true, y_pred_half_wrong)
        loss_all_wrong = losses.binary_cross_entropy(y_true, y_pred_all_wrong)

        with self.cached_session():
            self.assertAlmostEqual(loss_corr.eval(), 0., places=6)
            self.assertGreater(loss_all_wrong.eval(), loss_half_wrong.eval())

    def test_cross_entropy(self):
        """Test for cross_entropy()."""

        y_true = tf.convert_to_tensor([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])
        y_pred_corr = tf.convert_to_tensor([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])
        y_pred_half_wrong = tf.convert_to_tensor([
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        y_pred_all_wrong = tf.convert_to_tensor([
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
        ])

        loss_corr = losses.cross_entropy(y_true, y_pred_corr)
        loss_half_wrong = losses.cross_entropy(y_true, y_pred_half_wrong)
        loss_all_wrong = losses.cross_entropy(y_true, y_pred_all_wrong)

        with self.cached_session():
            self.assertAlmostEqual(loss_corr.eval(), 0., places=6)
            self.assertGreater(loss_all_wrong.eval(), loss_half_wrong.eval())

    def test_dice(self):
        """Test for dice()."""

        y_true = tf.convert_to_tensor([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])
        y_pred_corr = tf.convert_to_tensor([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])
        y_pred_half_wrong = tf.convert_to_tensor([
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
        y_pred_all_wrong = tf.convert_to_tensor([
            [0., 1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
        ])

        loss_corr = losses.dice(y_true, y_pred_corr)
        loss_half_wrong = losses.dice(y_true, y_pred_half_wrong)
        loss_all_wrong = losses.dice(y_true, y_pred_all_wrong)

        with self.cached_session():
            self.assertAlmostEqual(loss_corr.eval(), 0., places=6)
            self.assertGreater(loss_all_wrong.eval(), loss_half_wrong.eval())

    def test_cross_entropy_dice(self):
        """Test for cross_entropy_dice()."""

        y_true = tf.convert_to_tensor(np.random.uniform((3, 4, 5)))
        y_pred = tf.convert_to_tensor(np.random.uniform((3, 4, 5)))

        dice_loss = losses.dice(y_true, y_pred)
        cross_entropy_loss = losses.cross_entropy(y_true, y_pred)
        cross_entropy_dice_loss = losses.cross_entropy_dice(y_true, y_pred)

        with self.cached_session():
            self.assertEqual(dice_loss.eval() + cross_entropy_loss.eval(),
                             cross_entropy_dice_loss.eval())

    def test_cbinary_cross_entropy_dice(self):
        """Test for binary_cross_entropy_dice()."""

        y_true = tf.convert_to_tensor(np.random.uniform((3, 4, 5)))
        y_pred = tf.convert_to_tensor(np.random.uniform((3, 4, 5)))

        dice_loss = losses.dice(y_true, y_pred)
        binary_cross_entropy_loss = losses.binary_cross_entropy(y_true, y_pred)
        binary_cross_entropy_dice_loss = losses.binary_cross_entropy_dice(
            y_true, y_pred)

        with self.cached_session():
            self.assertEqual(
                dice_loss.eval() + binary_cross_entropy_loss.eval(),
                binary_cross_entropy_dice_loss.eval())


if __name__ == '__main__':
    tf.test.main()
