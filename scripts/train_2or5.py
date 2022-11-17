import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)




def main(argv):
    del argv  # Unused.

    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='330-')
    logging.info(f'workdir: {workdir}')

    config = FLAGS.config

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    app.run(main)