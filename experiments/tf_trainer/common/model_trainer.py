# coding=utf-8
# Copyright 2018 The Conversation-AI.github.io Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Model Trainer class.

This provides an abstraction of Keras and TF.Estimator, and is intended for use
in text classification models (although it may generalize to other kinds of
problems).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import comet_ml
import tensorflow as tf
import os
import os.path
import json
from typing import Dict, Any

from tf_trainer.common import dataset_input as ds
from tf_trainer.common import tfrecord_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import types
from tf_trainer.common import base_model
from tf_trainer import convai_config

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_path', None,
                           'Path to the training data TFRecord file.')
tf.app.flags.DEFINE_string('validate_path', None,
                           'Path to the validation data TFRecord file.')
tf.app.flags.DEFINE_string('model_dir', None,
                           "Directory for the Estimator's model directory.")
tf.app.flags.DEFINE_string('comet_api_key', None,
                           'String value of comet.ml key. Overrides config.')
tf.app.flags.DEFINE_string('comet_team_name', None,
                           'Name of comet team that tracks results. Overrides config.')
tf.app.flags.DEFINE_string('comet_project_name', None,
                           'Name of comet project that tracks results. Overrides config.')

tf.app.flags.mark_flag_as_required('train_path')
tf.app.flags.mark_flag_as_required('validate_path')
tf.app.flags.mark_flag_as_required('model_dir')


class ModelTrainer():
  """Model Trainer.

  Convenient way to run a text classification estimator, supporting comet.ml
  outputs.
  """

  def __init__(self, dataset: ds.DatasetInput,
               model: base_model.BaseModel) -> None:
    self._dataset = dataset
    self._model = model
    self._estimator = model.estimator(self._model_dir())

  # TODO(ldixon): consider early stopping. Currently steps is hard coded.
  def train_with_eval(self, steps, eval_period, eval_steps):
    """
    Args:
      steps: total number of batches to train for.
      eval_period: the number of steps between evaluations.
      eval_steps: the number of batches that are evaluated per evaulation.
    """
    experiment = self._setup_comet()
    num_itr = int(steps / eval_period)

    for _ in range(num_itr):
      self._estimator.train(
          input_fn=self._dataset.train_input_fn, steps=eval_period)
      metrics = self._estimator.evaluate(
          input_fn=self._dataset.validate_input_fn, steps=eval_steps)
      if experiment is not None:
        tf.logging.info('Logging metrics to comet.ml: {}'.format(metrics))
        experiment.log_multiple_metrics(metrics)
      tf.logging.info(metrics)

  def _setup_comet(self):
    if FLAGS.comet_api_key is not None:
      convai_config.comet_api_key = FLAGS.comet_api_key
    if FLAGS.comet_project_name is not None:
      convai_config.comet_project_name = FLAGS.comet_project_name
    if FLAGS.comet_team_name is not None:
      convai_config.comet_team_name = FLAGS.comet_team_name
    if not convai_config.comet_api_key:
      return None
    experiment = comet_ml.Experiment(
        api_key=convai_config.comet_api_key,
        project_name=convai_config.comet_project_name,
        team_name=convai_config.comet_team_name,
        auto_param_logging=False,
        parse_args=False)
    experiment.log_parameter('train_path', FLAGS.train_path)
    experiment.log_parameter('validate_path', FLAGS.validate_path)
    experiment.log_parameter('model_dir', self._model_dir())
    experiment.log_multiple_params(self._model.hparams().values())
    return experiment

  def _model_dir(self):
    """Get Model Directory.

    Used to scope logs to a given trial (when hyper param tuning) so that they
    don't run over each other. When running locally it will just use the passed
    in model_dir.
    """
    return os.path.join(
        FLAGS.model_dir,
        json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get(
            'trial', ''))
