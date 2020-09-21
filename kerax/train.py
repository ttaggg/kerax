"""Train file."""

import os
import time

from absl import app
from absl import logging

from kerax import flags
from kerax.models import model_hub
from kerax.utils import general_utils as gutils

FLAGS = flags.FLAGS


def main(_):
    """Run pipeline."""

    # Set config, output directory, logging levels and random seed.
    config = gutils.get_config(FLAGS.config)
    gutils.initialize(output_dir=FLAGS.output_dir, seed=FLAGS.random_seed)

    # Create model.
    if FLAGS.load_saved_model:
        model, initial_epoch = gutils.get_model(FLAGS.load_saved_model)
    else:
        model = model_hub.create_model(config['model'])
        initial_epoch = 0

    model.summary()
    # Compile model.
    train_config = config['training']
    model.compile(
        optimizer=gutils.get_optimizer(train_config['optimizer']),
        loss=gutils.get_losses(train_config['loss_function']),
        metrics=gutils.get_metrics(train_config['metrics']))
    logging.info('Model is compiled.')

    # Load dataset,
    data_config = config['data']
    loader = gutils.get_loader(data_config)
    train_generator, test_generator = loader.generators(
        train_config['batch_size'])
    logging.info('Generators are created.')

    # Fit generator.
    callbacks = gutils.get_callbacks(test_generator, train_config['callbacks'],
                                     FLAGS.output_dir)
    model.fit_generator(
        generator=train_generator,
        epochs=train_config['num_epochs'],
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        validation_data=test_generator,
        workers=FLAGS.workers,
        use_multiprocessing=FLAGS.use_multiprocessing,
        verbose=2)
    logging.info('Training is finished.')

    # Save weights and model.
    current_time = int(time.time())
    saved_dir = os.path.join(FLAGS.output_dir, 'saved_model')
    os.makedirs(saved_dir, exist_ok=True)
    model.save_weights(
        filepath=os.path.join(saved_dir, f'{current_time}.weights'))
    model.save(filepath=os.path.join(saved_dir, f'{current_time}.model'))


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)
