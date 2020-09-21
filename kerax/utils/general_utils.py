"""General utils."""

import os
import re
import random
import runpy
import collections

import keras_tqdm
import numpy as np
import tensorflow as tf
from absl import logging
from keras import callbacks as keras_callbacks
from keras import models
from keras import optimizers
from PIL import Image

from kerax.utils import callbacks
from kerax.utils import image_utils
from kerax.utils import losses
from kerax.utils import metrics
from kerax.loaders import cityscapes_data_loader
from kerax.loaders import mauto_data_loader
from kerax.loaders import severstal_data_loader


def initialize(output_dir, seed):
    """Initilialize output directory and logging levels."""
    # Initialize output directory.
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Set logging levels.
    logging.get_absl_handler().use_absl_log_file('log_file', log_dir)
    logging.set_stderrthreshold(logging.INFO)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Fix seeds.
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    logging.info(f'Random seed is {seed}.')


def _dict_to_defaultdict(config):
    """Recursively convert all dicts to defaultdicts."""
    if not isinstance(config, collections.defaultdict):
        if isinstance(config, dict):
            new_config = collections.defaultdict(collections.defaultdict)
            new_config.update(
                {k: _dict_to_defaultdict(v) for k, v in config.items()})
            return new_config
    return config


def check_required_config_fields(config, only_predict):
    """Check whether all required fields exist.

    Args:
        config: Dictionary with settings.
        only_predict: Boolean, for prediction only data config is required.
    Raises:
        AssertionError if one of the data, training, model is missing.
    """

    # Check data config requirements.
    assert 'data' in config, f'data config is required.'
    for req in {'data_path', 'labels_path', 'task'}:
        assert req in config['data'], f'{req} in data config is required.'

    # Check model and training config.
    if not only_predict:
        for req in {'model', 'training'}:
            assert req in config, f'{req} config is required.'
        for req in {'model_name', 'input_shape', 'num_classes'}:
            assert req in config['model'], f'{req} in model config is required.'
        for req in {'batch_size', 'num_epochs', 'loss_function'}:
            assert req in config[
                'training'], f'{req} in training config is required.'


def get_config(config_file, only_predict=False):
    """Parse config.

    Args: config_file: path to config file,
        consists of config dictionaries,
        data, model, training are required.
    Returns:
        config: dictionary of config dictionaries.
    Raises:
        FileNotFoundError if config_file was not found.
    """
    if os.path.exists(config_file):
        config = runpy.run_path(config_file)
        check_required_config_fields(config, only_predict)
        # TODO(ttaggg): make config immutable.
        return _dict_to_defaultdict(config)
    raise FileNotFoundError(f'File {config_file} was not found.')


def get_optimizer(optimizer_config):
    """Choose optimizer.

    Args:
        optimizer_config: Dictionary.
    Returns:
        keras.optimizers.Adam or RMSprop.
    Raises:
        ValueError if optimizer name is not one of
            adam or rmsprop.
    """
    optimizer = optimizer_config.get('optimizer_name', 'adam')
    learning_rate = optimizer_config.get('learning_rate', 1e-3)
    optimizer_params = optimizer_config.get('params', {})

    logging.info(f'Optimizer: {optimizer}.')
    if optimizer == 'adam':
        return optimizers.Adam(lr=learning_rate, **optimizer_params)
    if optimizer == 'rmsprop':
        return optimizers.RMSprop(**optimizer_params)
    raise ValueError('Only "adam" and "rmsprop" are supported,'
                     f'given: {optimizer}.')


def get_losses(loss_name):
    """Choose loss function.

    Args:
        loss_name: String, name of loss function.
    Returns:
        loss_fn: Function from utils.losses.
    Raises:
        ValueError if loss_name is not one of the
            binary_cross_entropy, binary_cross_entropy_dice,
            cross_entropy, cross_entropy_dice, huber_loss,
            mean_squared_error, mean_absolute_error, l1_l2_loss.
    """
    logging.info(f'Loss_function: {loss_name}.')

    losses_dict = {
        'binary_cross_entropy': losses.binary_cross_entropy,
        'binary_cross_entropy_dice': losses.binary_cross_entropy_dice,
        'cross_entropy': losses.cross_entropy,
        'cross_entropy_dice': losses.cross_entropy_dice,
        'dice': losses.dice,
        'huber_loss': losses.huber_loss,
        'mean_squared_error': losses.mean_squared_error,
        'mean_absolute_error': losses.mean_absolute_error,
        'l1_l2_loss': losses.l1_l2_loss,
    }
    if loss_name not in losses_dict:
        raise ValueError(f'Only losses from: {list(losses_dict.keys())}'
                         f' are supported, given: {loss_name}.')
    loss_fn = losses_dict[loss_name]
    return loss_fn


def get_metrics(metrics_names):
    """Return metrics.
    Args:
        metrics_names: List of metrics names to use.
    Returns:
        metrics_fns: List of metrics functions
            from utils.metrics or string with
            supported by Keras metric.
    """
    if metrics_names:
        logging.info(f'Metrics: {", ".join(metrics_names)}.')

    metrics_dict = {
        'dice': metrics.dice,
        'mcc': metrics.mcc,
        'mse': metrics.mse,
        'mae': metrics.mae,
        'accuracy': 'accuracy',
    }
    metrics_fns = [metrics_dict[name] for name in metrics_names]
    return metrics_fns


def _get_custom_objects():
    """Get custom objects for loading checkpoint."""
    return {
        'binary_cross_entropy': losses.binary_cross_entropy,
        'bce_dice': losses.binary_cross_entropy_dice,
        'cross_entropy': losses.cross_entropy,
        'ce_dice': losses.cross_entropy_dice,
        'dice': metrics.dice,
        'mcc': metrics.mcc,
    }


def get_postprocessor(task):
    # TODO(ttaggg): do it in normal way.

    if task.endswith('segmentation'):

        def postprocess_images(images, predictions, batch, generator,
                               output_dir):

            masks = np.argmax(predictions, axis=-1)
            # Assign each class different color.
            if generator.colormap is not None:
                colormap = generator.colormap
            else:
                colormap = image_utils.create_label_colormap(masks.shape[-1])

            for i, (image, mask) in enumerate(zip(images, masks)):

                # If image is grayscale, convert to kinda-RGB for logging.
                if image.shape[-1] == 1:
                    image = np.concatenate([image] * 3, axis=-1)

                mask = colormap[mask]

                # If we normalized image use that means and stds.
                if generator.normalize_shift is not None:
                    means, stds = generator.normalize_shift
                    image = np.clip((image * stds) + means, 0., 1.)

                # Upper part of the final image is ground truth.
                # Middle part of the final image is image.
                # Lower part of the final image is predicted mask.
                log_image = np.concatenate([image, mask], axis=0)
                image = Image.fromarray((255. * log_image).astype(np.uint8))
                filename = os.path.join(output_dir, f'{batch}_{i}.jpg')
                image.save(filename)
                logging.info(f'Saving to {filename}.')

        postprocess_fn = postprocess_images

    elif task.endswith('classification'):

        def postprocess_labels(image, predictions, batch, generator,
                               output_dir):
            del image  # Unused.
            del batch  # Unused.
            del generator  # Unused.
            outfile = open(os.path.join(output_dir, 'results.txt'), 'a')
            labels = np.argmax(predictions, axis=-1)
            for label in labels:
                outfile.write(str(label))
                outfile.write('\n')

        postprocess_fn = postprocess_labels

    elif task.endswith('regression'):

        def postprocess_labels(image, predictions, batch, generator,
                               output_dir):
            del image  # Unused.
            del batch  # Unused.
            del generator  # Unused.
            outfile = open(os.path.join(output_dir, 'results.txt'), 'a')
            for label in predictions:
                outfile.write(str(label[0]))
                outfile.write('\n')

        postprocess_fn = postprocess_labels

    else:
        raise ValueError('Only *_segmentation and *_classification tasks '
                         f'are supported, given: {task}.')

    return postprocess_fn


def get_model(model_path, compile_model=False):
    """Load checkpoint from model directory.

    Args:
        model_path: String, path to saved mode or "resume" if model is
            in the FLAGS.output_dir/saved directory.
            Path to saved model could be either directory with
            checkpoints: in this case last one is used, or
            path to particular checkpoint.
        compile_model: Bool, optional: whether to compile model.
    Returns:
        model: Keras model.
        epoch: Integer, current epoch.
    Raises:
        FileNotFoundError if model_path cannot be found or empty.
    """

    if model_path.lower() == 'resume':
        model_path = os.path.join(os.getcwd(), 'saved')

    if not os.path.exists(model_path):
        raise FileNotFoundError('Asked to resume training,'
                                'but no "saved" directory was found.')

    if model_path.endswith('hdf5'):
        checkpoint = model_path
    elif os.path.isdir(model_path):
        checkpoints = os.listdir(model_path)
        if not checkpoints:
            raise FileNotFoundError(f'{model_path} directory is empty.')
        # Find the latest checkpoint in the directory.
        checkpoint = sorted(
            checkpoints, key=lambda x: int(x.split('.')[-2]))[-1]
        model_path = os.path.join(model_path, checkpoint)
    else:
        raise ValueError('Please give path to valid checkpoint, or '
                         f'directory with checkpoints, given: {model_path}.')

    epoch = int(re.match(r'(.*)chpt.(\d+).hdf5', checkpoint).group(2))
    model = models.load_model(
        model_path, custom_objects=_get_custom_objects(), compile=compile_model)

    logging.info('Model is loaded.')
    logging.info(f'Checkpoint: {checkpoint}.')
    return model, epoch


def get_loader(config):
    """Get appropriate loader."""
    if config['task'].startswith('severstal'):
        return severstal_data_loader.SeverstalDataLoader(config)
    if config['task'].startswith('cityscapes'):
        return cityscapes_data_loader.CityscapesDataLoader(config)
    if config['task'].startswith('mauto'):
        return mauto_data_loader.MAutoDataLoader(config)
    raise ValueError('Currently only cityscapes, mauto and severstal tasks '
                     f'are supported, given: {config["task"]}.')


def get_callbacks(generator, callbacks_config, output_dir):
    """Get callbacks.

    Args:
        generator: Keras test_generator.
        callbacks_config: Dictionary.
            Contains optional elements:
                custom_callbacks: dict of dicts: names
                    of classes from utils.callbacks and their params.
                monitor: String, value of argument monitor in
                    Keras standard callbacks.
                monitor_mod: String, value of argument monitor_mode in
                    Keras standard callbacks.
                patience: Integer, value of argument patience in
                    Keras standard callbacks.
    Returns:
        callbacks_list: List of callbacks objects to use.
    """

    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks_dict = {
        'ImageLogger':
            callbacks.ImageLogger,
        'SegmentationDiceEpochCallback':
            callbacks.SegmentationDiceEpochCallback,
        'MultiClassifierEpochCallback':
            callbacks.MultiClassifierEpochCallback,
        'RegressionEpochCallback':
            callbacks.RegressionEpochCallback,
    }

    callbacks_list = []

    custom_callbacks = callbacks_config.get('custom_callbacks', {})
    for callback in custom_callbacks:
        arguments = custom_callbacks[callback]
        callbacks_list.append(callbacks_dict[callback](
            generator=generator, log_dir=log_dir, prefix='val', **arguments))

    monitor = callbacks_config.get('monitor', 'val_loss')
    monitor_mode = callbacks_config.get('monitor_mode', 'auto')
    lr_factor = callbacks_config.get('lr_factor', 0.5)
    patience = callbacks_config.get('patience', 2)
    tb_update_freq = callbacks_config.get('tb_update_freq', 500)
    chpt_name = os.path.join(ckpt_dir, 'chpt.{epoch:02d}.hdf5')

    callbacks_list.append(
        keras_callbacks.ModelCheckpoint(
            filepath=chpt_name,
            monitor=monitor,
            mode=monitor_mode,
            verbose=1,
            save_best_only=False))

    callbacks_list.append(
        keras_callbacks.ReduceLROnPlateau(
            factor=lr_factor,
            monitor=monitor,
            mode=monitor_mode,
            patience=patience,
            min_lr=0.5e-6,
            verbose=1))

    # Value of update_freq should divide 'update_freq' of other
    # custom callbacks to see them on tensorboard.
    callbacks_list.append(
        callbacks.CustomTensorBoardCallback(
            log_dir=log_dir,
            histogram_freq=0,
            write_images=True,
            write_graph=True,
            update_freq=tb_update_freq))

    callbacks_list.append(keras_tqdm.TQDMCallback())
    callbacks_list.append(keras_callbacks.BaseLogger())

    return callbacks_list
