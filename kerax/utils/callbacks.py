"""Custom callbacks for tensorboard."""
# TODO(ttaggg): refactor this file.

import os

import numpy as np
import tensorflow as tf
from absl import logging
from keras import callbacks as keras_callbacks
from scipy import signal
from sklearn import metrics

from kerax.utils import image_utils


def _tf_image_summary(name, image):
    return tf.compat.v1.Summary.Value(tag=name, image=image)


def _np_dice_score(y_true, y_pred, eps=1e-8):
    """Calculate dice coefficient."""
    if np.sum(y_true) == 0 and np.sum(y_pred) != 0:
        return 0
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + eps)


def _write_predictions(predictions, filename):
    with open(filename, 'w') as outfile:
        for x in predictions:
            outfile.write(str(x))
            outfile.write('\n')


class ImageLogger(keras_callbacks.Callback):
    """Logging images to tensorboard."""

    def __init__(self,
                 generator,
                 prefix,
                 log_dir,
                 batches_to_log=8,
                 update_freq=1000):
        super().__init__()
        self._generator = generator
        self._log_dir = os.path.join(log_dir, prefix, 'images')
        self._batches_to_log = batches_to_log
        self._update_freq = update_freq
        self._prefix = prefix
        self._writer = tf.compat.v1.summary.FileWriter(self._log_dir)
        self._batches_seen = 0

        _, masks = self._generator[0]

        if self._generator.colormap is not None:
            self._colormap = self._generator.colormap
        else:
            self._colormap = image_utils.create_label_colormap(masks.shape[-1])

    def on_batch_end(self, batch, logs=None):
        self._batches_seen += 1
        if self._batches_seen % self._update_freq == 0:
            self._write_images_summary(self._batches_seen)

    def _write_images_summary(self, batch):
        """Write down images summary for tensorboard."""

        summary_strs = []
        for sample, (images, ground_masks) in enumerate(self._generator):
            if sample >= self._batches_to_log:
                break

            predictions = self.model.predict_on_batch(images)
            masks = np.argmax(predictions, axis=-1)
            ground_masks = np.argmax(ground_masks, axis=-1)

            summary_str = self._create_batch_image_summary(
                sample, images, ground_masks, masks)
            summary_strs.extend(summary_str)

        self._writer.add_summary(
            tf.compat.v1.Summary(value=summary_strs), global_step=batch)

    def _create_batch_image_summary(self, sample, images, ground_masks, masks):
        """Colorify images and masks, create list of TF summaries."""

        summary_str = []
        for i, (image, true_mask,
                mask) in enumerate(zip(images, ground_masks, masks)):
            # If image is grayscale, convert to kinda-RGB for logging.
            if image.shape[-1] == 1:
                image = np.concatenate([image] * 3, axis=-1)

            # If we normalized image use that means and stds.
            if self._generator.normalize_shift is not None:
                means, stds = self._generator.normalize_shift
                image = np.clip((image * stds) + means, 0., 1.)

            # Assign each class different color.
            true_mask = self._colormap[true_mask]
            mask = self._colormap[mask]
            # Upper part of the final image is ground truth.
            # Middle part of the final image is image.
            # Lower part of the final image is predicted mask.
            log_image = np.concatenate([true_mask, image, mask], axis=0)
            summary_str.append(
                _tf_image_summary(f'{self._prefix}/plot/{sample}/{i}',
                                  image_utils.image_to_summary(log_image)))
        return summary_str


class SegmentationDiceEpochCallback(keras_callbacks.Callback):
    """Dice metric for segmentation task."""

    def __init__(self,
                 generator,
                 prefix,
                 log_dir=None,
                 batches_to_log=None,
                 update_freq=1000,
                 cut_background_index=None):
        del log_dir  # Unused.
        super().__init__()
        self._generator = generator
        self._batches_to_log = batches_to_log
        self._update_freq = update_freq
        self._prefix = prefix
        self._cut_background_index = cut_background_index
        self._batches_seen = 0

    def on_epoch_end(self, epoch, logs=None):
        self._evaluate_metrics(self._batches_seen, logs)

    def on_batch_end(self, batch, logs=None):
        self._batches_seen += 1
        if self._batches_seen % self._update_freq == 0:
            self._evaluate_metrics(self._batches_seen, logs)

    def _evaluate_metrics(self, batch, logs):
        """Evaluate metrics on the whole set."""
        prefix = self._prefix
        dice_scores = []
        for i, (images, ground_masks) in enumerate(self._generator):
            if self._batches_to_log and i >= self._batches_to_log:
                break

            results = self.model.predict(images)
            ground_masks, masks = self._postprocess_masks(ground_masks, results)
            # Calculate dice metric for all pairs.
            for mask, ground_mask in zip(masks, ground_masks):
                dice_scores.append(_np_dice_score(ground_mask, mask))

        mean_dice = np.mean(dice_scores)
        logs[f'{self._prefix}_dice_epoch'] = mean_dice
        logging.info(f'Step: {batch}, {prefix}_dice_epoch: {mean_dice: .4f}.')

    def _postprocess_masks(self, ground_masks, results):
        """Create mask from logits, cut masks if necessary."""
        class_count = ground_masks.shape[-1]
        results = np.argmax(results, axis=-1)
        masks = np.eye(class_count)[results]
        # Cut out background layer that takes the largest part of the image.
        if self._cut_background_index is not None:
            if self._cut_background_index == 'first':
                masks = masks[..., 1:]
                ground_masks = ground_masks[..., 1:]
            elif self._cut_background_index == 'last':
                masks = masks[..., :-1]
                ground_masks = ground_masks[..., :-1]
            else:
                raise ValueError(
                    'Supported cut_background_index: "first" and "last"'
                    f'given: {self._cut_background_index}')
        return ground_masks, masks


class MultiClassifierEpochCallback(keras_callbacks.Callback):
    """Metrics for classification task."""

    def __init__(self,
                 generator,
                 prefix,
                 threshold=0.5,
                 update_freq=1000,
                 log_dir=None):
        del log_dir  # Unused.
        super().__init__()
        self._generator = generator
        self._update_freq = update_freq
        self._prefix = prefix
        self._threshold = threshold
        self._batches_seen = 0

    def on_epoch_end(self, epoch, logs=None):
        self._evaluate_metrics(self._batches_seen, logs)

    def on_batch_end(self, batch, logs=None):
        self._batches_seen += 1
        if self._batches_seen % self._update_freq == 0:
            self._evaluate_metrics(self._batches_seen, logs)

    def _evaluate_metrics(self, batch, logs):
        """Evaluate metrics on the whole set."""
        logging.info(f'Step: {batch}.')

        labels = []
        predictions = []
        for image, label in self._generator:
            prediction = self.model.predict(image)
            labels.append(label)
            predictions.append(prediction)

        labels = np.concatenate(labels, axis=0).astype(np.int32)
        predictions = (np.concatenate(predictions, axis=0) >
                       self._threshold).astype(np.int32)

        set_of_metrics = {
            metrics.f1_score,
            metrics.precision_score,
            metrics.recall_score,
        }
        logging.info(f'Step: {batch}.')
        for metric_fn in set_of_metrics:
            metric = metric_fn(labels, predictions, average='micro')
            metric_name = f'{self._prefix}_{metric_fn.__name__}'
            logs[metric_name] = metric
            logging.info(f'{metric_name}: {metric: .4f}')


class RegressionEpochCallback(keras_callbacks.Callback):
    """Metrics for classification task."""

    def __init__(self,
                 generator,
                 prefix,
                 update_freq=1000,
                 rolling_mean=None,
                 log_dir=None):
        super().__init__()
        self._generator = generator
        self._update_freq = update_freq
        self._prefix = prefix
        self._batches_seen = 0
        self._rolling_mean = rolling_mean
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        self._evaluate_metrics(self._batches_seen, logs)

    def on_batch_end(self, batch, logs=None):
        self._batches_seen += 1
        if self._batches_seen % self._update_freq == 0:
            self._evaluate_metrics(self._batches_seen, logs)

    def _evaluate_metrics(self, batch, logs):
        """Evaluate metrics on the whole set."""

        labels = []
        predictions = []
        for image, label in self._generator:
            prediction = self.model.predict(image)
            labels.extend(label)
            predictions.extend(prediction)

        labels = np.array(labels).reshape(-1)
        predictions = np.array(predictions).reshape(-1)

        _write_predictions(
            predictions,
            os.path.join(self._log_dir, f'raw_predictions_{batch}'),
        )
        _write_predictions(
            labels,
            os.path.join(self._log_dir, f'labels_{batch}'),
        )

        mse = metrics.mean_squared_error(labels, predictions)
        logs[f'{self._prefix}_mse_epoch'] = mse
        logging.info(f'Step: {batch}; {self._prefix}_mse_epoch: {mse}')

        if self._rolling_mean:
            smooth_preds = self._moving_average(predictions, self._rolling_mean)
            smooth_mse = metrics.mean_squared_error(labels, smooth_preds)
            logs[f'{self._prefix}_smooth_mse_epoch'] = smooth_mse
            logging.info(
                f'Step: {batch}; {self._prefix}_smooth_mse_epoch: {smooth_mse}')
            _write_predictions(
                smooth_preds,
                os.path.join(self._log_dir, f'smooth_preds_{batch}'))

        logging.info('Labels: %s', labels)
        logging.info('Predictions: %s', predictions)
        logging.info('Smooth predictions: %s', smooth_preds)

    def _moving_average(self, data, rolling_mean=41):
        """Smooth labels by using rolling mean."""
        data[data < 0.] = 0.
        data = signal.savgol_filter(
            data, window_length=rolling_mean, polyorder=1)
        data[data < 0.] = 0.
        return data


class CustomTensorBoardCallback(keras_callbacks.TensorBoard):
    """Custom tensorboard callback.

    Override on_batch_end():
        1. Make steps on tensorboard to be steps, not samples seen.
        2. Synchronize writing to tensorboard with other callbacks.
    Override on_epoch_end():
        1. Write down all values from other callbacks on the end of
            the epoch, regarless of update_freq.
    """

    def __init__(self, *args, **kwargs):
        self._batches_seen = 0
        if not isinstance(kwargs.get('update_freq', None), int):
            logging.warning('CustomTensorBoardCallback only works with '
                            'integer values for "update_freq".')
        super().__init__(*args, **kwargs)

    def on_batch_end(self, batch, logs=None):
        """Original code:

        if self.update_freq != 'epoch':
            self.samples_seen += 1  # logs['size'] in original code.
            samples_seen_since = (
                self.samples_seen - self.samples_seen_at_last_write)
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen
        """
        self._batches_seen += 1
        if self.update_freq != 'epoch':
            if self._batches_seen % self.update_freq == 0:
                self._write_logs(logs, self._batches_seen)

    def on_epoch_end(self, epoch, logs=None):
        if self.update_freq != 'epoch':
            self._write_logs(logs, self._batches_seen)
