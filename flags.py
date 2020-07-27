"""All flags that are used in the pipeline."""

import os

from absl import flags

flags.DEFINE_string('output_dir', os.getcwd(), 'Path to output directory.')
flags.DEFINE_string('config', None, 'Path to train settings.')
flags.DEFINE_string('load_saved_model', None, 'Which checkpoint to use.')
flags.DEFINE_integer('random_seed', 42, 'Random seed value.')
flags.DEFINE_integer('workers', 8, 'Number of workers.')
flags.DEFINE_bool('use_multiprocessing', True, 'Whether use multiprocessing.')

FLAGS = flags.FLAGS


def mark_flags_as_required(*args, **kwargs):
    """"Override absl.flags.mark_flags_as_required """
    flags.mark_flags_as_required(*args, **kwargs)
