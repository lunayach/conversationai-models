"""Tensorflow Reptile.

Copied with modifications from:
https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/reptile.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import layers
from tf_trainer.common import base_model
from typing import Set


class TFReptileModel(object):
  """TF Reptile.

  TF implementation of Reptile (https://arxiv.org/abs/1803.02999).
  Inputs should be sequences of word embeddings.
  """

  def __init__(self,
               session,
               variables=None,
               transductive=False,
               pre_step_op=None):
    self.session = session
    self._model_state = VariableState(
        self.session, variables or tf.trainable_variables())
    self._full_state = VariableState(
        self.session,
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    self._transductive = transductive
    self._pre_step_op = pre_step_op


  def train_step(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement,
                 meta_step_size,
                 meta_batch_size):
    """
    Perform a Reptile training step.
    Args:
      dataset: a sequence of data classes, where each data
        class has a sample(n) method.
      input_ph: placeholder for a batch of samples.
      label_ph: placeholder for a batch of labels.
      minimize_op: TensorFlow Op to minimize a loss on the
        batch specified by input_ph and label_ph.
      num_classes: number of data classes to sample.
      num_shots: number of examples per data class.
      inner_batch_size: batch size for every inner-loop
        training iteration.
      inner_iters: number of inner-loop iterations.
      replacement: sample with replacement.
      meta_step_size: interpolation coefficient.
      meta_batch_size: how many inner-loops to run.
    """
    old_vars = self._model_state.export_variables()
    new_vars = []
    for _ in range(meta_batch_size):
      mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
      for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
        inputs, labels = zip(*batch)
        if self._pre_step_op:
          self.session.run(self._pre_step_op)
        self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
      new_vars.append(self._model_state.export_variables())
      self._model_state.import_variables(old_vars)
    new_vars = average_vars(new_vars)
    self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))


    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
      """
      Run a single evaluation of the model.
      Samples a few-shot learning task and measures
      performance.
      Args:
        dataset: a sequence of data classes, where each data
          class has a sample(n) method.
        input_ph: placeholder for a batch of samples.
        label_ph: placeholder for a batch of labels.
        minimize_op: TensorFlow Op to minimize a loss on the
          batch specified by input_ph and label_ph.
        predictions: a Tensor of integer label predictions.
        num_classes: number of data classes to sample.
        num_shots: number of examples per data class.
        inner_batch_size: batch size for every inner-loop
          training iteration.
        inner_iters: number of inner-loop iterations.
        replacement: sample with replacement.
      Returns:
        The number of correctly predicted samples.
          This always ranges from 0 to num_classes.
      """
      train_set, test_set = _split_train_test(
        _sample_mini_dataset(dataset, num_classes, num_shots+1))
      old_vars = self._full_state.export_variables()
      for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
        inputs, labels = zip(*batch)
        if self._pre_step_op:
          self.session.run(self._pre_step_op)
        self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
      test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
      num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
      self._full_state.import_variables(old_vars)
      return num_correct


    def _test_predictions(self, train_set, test_set, input_ph, predictions):
      if self._transductive:
          inputs, _ = zip(*test_set)
          return self.session.run(predictions, feed_dict={input_ph: inputs})
      res = []
      for test_sample in test_set:
          inputs, _ = zip(*train_set)
          inputs += (test_sample[0],)
          res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
      return res