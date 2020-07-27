"""Prediction file."""

from absl import app

import flags
from utils import general_utils as gutils

FLAGS = flags.FLAGS


def main(_):

    # Set config, output directory, logging levels and random seed.
    config = gutils.get_config(FLAGS.config, only_predict=True)
    gutils.initialize(output_dir=FLAGS.output_dir, seed=FLAGS.random_seed)
    data_config = config['data']

    # Load model.
    model, _ = gutils.get_model(FLAGS.load_saved_model)

    # Load dataset.
    loader = gutils.get_loader(data_config)
    _, test_generator = loader.generators()
    postprocess_fn = gutils.get_postprocessor(data_config['task'])

    for batch, (images, _) in enumerate(test_generator):
        predictions = model.predict_on_batch(images)
        postprocess_fn(images, predictions, batch, FLAGS.output_dir)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'load_saved_model'])
    app.run(main)
