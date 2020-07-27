"""Tests for metrics.py"""

import numpy as np
import tensorflow as tf
from sklearn import metrics as sklearn_metrics

from utils import metrics


class MetricsTest(tf.test.TestCase):

    def setUp(self):
        np.random.seed(42)
        tf.compat.v1.set_random_seed(42)

    def test_dice(self):
        """Test for dice()."""

        # Manual check.
        y_true = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1., 0.]])
        y_pred = np.array([[0, 0, 0.9, 0], [0, 0, 0.1, 0], [1, 1, 0.1, 1.]])

        # Intersection = 0.9 + 0.1 + 0.1. = 1.1
        # Numerator: 2 * 1.1 + 1 = 3.2
        # Denominator: 3 + 4.1 + 1 = 8.1
        # Result: 3.2 / 8.1 = 0.39506172839
        dice = metrics.dice(y_true, y_pred)
        with self.cached_session():
            self.assertAlmostEqual(dice.eval(), 0.39506172839)

        # Sanity check.
        y_true = np.array([[0., 0., 1., 0.], [1., 0., 1., 0.]])
        y_pred = np.array([[0., 0., 1., 0.], [1., 0., 1., 0.]])

        dice = metrics.dice(y_true, y_pred)
        with self.cached_session():
            self.assertAlmostEqual(dice.eval(), 1.)

        # Sanity check.
        y_true = np.array([[0., 0., 1., 0.], [1., 0., 1., 0.]])
        y_pred = np.array([[1., 1., 0., 1.], [0., 1., 0., 1.]])

        dice = metrics.dice(y_true, y_pred, smooth=0.0)
        with self.cached_session():
            self.assertAlmostEqual(dice.eval(), 0.)

        # Check on random integers data.
        y_true = np.random.randint(2, size=100)
        y_pred = np.random.randint(2, size=100)
        jaccard_index = sklearn_metrics.jaccard_score(y_true, y_pred)
        expected_dice = 2 * jaccard_index / (1 + jaccard_index)

        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        dice = metrics.dice(y_true, y_pred, smooth=0.)

        with self.cached_session():
            self.assertAlmostEqual(dice.eval(), expected_dice)

    def test_mcc(self):
        """Test for dice()."""

        # Check on random integers data.
        y_true = np.random.randint(2, size=100)
        y_pred = np.random.randint(2, size=100)
        expected_mcc = sklearn_metrics.matthews_corrcoef(y_true, y_pred)

        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        mcc = metrics.mcc(y_true, y_pred)

        with self.cached_session():
            self.assertAlmostEqual(mcc.eval(), expected_mcc)


if __name__ == '__main__':
    tf.test.main()
