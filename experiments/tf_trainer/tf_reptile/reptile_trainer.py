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

Copied with modifications from:
https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/train.py
https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/eval.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path
import six

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export.export_output import PredictOutput
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.lib.io import file_io

from tf_trainer.common import base_model
from tf_trainer.common import dataset_input as ds

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', None,
                           "Directory for the model.")
tf.app.flags.DEFINE_integer(
    'n_export', -1, 'Number of models to export.'
    'If =-1, only the best checkpoint (wrt specified eval metric) is exported.'
    'If =1, only the last checkpoint is exported.'
    'If >1, we export `n_export` evenly-spaced checkpoints.')
tf.app.flags.DEFINE_string('key_name', 'comment_key',
                           'Name of a pass-thru integer id for batch scoring.')

tf.app.flags.mark_flag_as_required('model_dir')


class ReptileTrainer(object):
  """Model Trainer."""

  def __init__(self,
               dataset: ds.DatasetInput,
               reptile: TFReptileModel,
               sess: ) -> None:
    self.dataset = dataset
    self.sess = sess
    self.reptile = reptile


  def train(self,
            model,
            train_set,
            test_set,
            save_dir,
            num_classes=5,
            num_shots=5,
            inner_batch_size=5,
            inner_iters=20,
            replacement=False,
            meta_step_size=0.1,
            meta_step_size_final=0.1,
            meta_batch_size=1,
            meta_iters=400000,
            eval_inner_batch_size=5,
            eval_inner_iters=50,
            eval_interval=10,
            weight_decay_rate=1,
            time_deadline=None,
            train_shots=None,
            transductive=False,
            log_fn=print):
    """
    Train a model on a dataset.
    """
      if not os.path.exists(save_dir):
          os.mkdir(save_dir)
      saver = tf.train.Saver()
      reptile = self.reptile(self.sess,
                           transductive=transductive,
                           pre_step_op=weight_decay(weight_decay_rate))
      accuracy_ph = tf.placeholder(tf.float32, shape=())
      tf.summary.scalar('accuracy', accuracy_ph)
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
      test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
      tf.global_variables_initializer().run()
      sess.run(tf.global_variables_initializer())
      for i in range(meta_iters):
          frac_done = i / meta_iters
          cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
          reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                             num_classes=num_classes, num_shots=(train_shots or num_shots),
                             inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                             replacement=replacement,
                             meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
          if i % eval_interval == 0:
              accuracies = []
              for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                  correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                             model.minimize_op, model.predictions,
                                             num_classes=num_classes, num_shots=num_shots,
                                             inner_batch_size=eval_inner_batch_size,
                                             inner_iters=eval_inner_iters, replacement=replacement)
                  summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes})
                  writer.add_summary(summary, i)
                  writer.flush()
                  accuracies.append(correct / num_classes)
              log_fn('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))
          if i % 100 == 0 or i == meta_iters-1:
              saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
          if time_deadline is not None and time.time() > time_deadline:
              break

  def eval(self,
           model,
           dataset,
           num_classes=5,
           num_shots=5,
           eval_inner_batch_size=5,
           eval_inner_iters=50,
           replacement=False,
           num_samples=10000,
           transductive=False,
           weight_decay_rate=1):
      """
      Evaluate a model on a dataset.
      """
      reptile = self.reptile(self.sess,
                           transductive=transductive,
                           pre_step_op=weight_decay(weight_decay_rate))
      total_correct = 0
      for _ in range(num_samples):
          total_correct += reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                            model.minimize_op, model.predictions,
                                            num_classes=num_classes, num_shots=num_shots,
                                            inner_batch_size=eval_inner_batch_size,
                                            inner_iters=eval_inner_iters, replacement=replacement)
      return total_correct / (num_samples * num_classes)

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