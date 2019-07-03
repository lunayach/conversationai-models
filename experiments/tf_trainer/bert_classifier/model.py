"""Tensorflow Estimator using TF Hub universal sentence encoder."""
#TODO(nthain): Maybe import os and manually set the cache file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
from tf_trainer.common import base_model
from typing import List

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
# TODO: Add validation
tf.app.flags.DEFINE_float('learning_rate', 0.00003,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_float('dropout_rate', 0.15,
                          'The dropout rate to use during training.')
tf.app.flags.DEFINE_string(
    'model_spec',
    'gs://conversationai-models/resources/tfhub/bert_uncased_L-12_H-768_A-12-1/5a395eafef2a37bd9fc55d7f6ae676d2a134a838',
    'The url of the TF Hub sentence encoding module to use.')
# TODO: Wire flag in 
tf.app.flags.DEFINE_bool('trainable', False,
                         'What to pass for the TF Hub trainable parameter.')


class BERTClassifierModel(base_model.BaseModel):

  @staticmethod
  def hparams():
    hparams = tf.contrib.training.HParams(
        learning_rate=FLAGS.learning_rate,
        dropout_rate=FLAGS.dropout_rate)
    return hparams

  def estimator(self, model_dir):
    num_labels = 2 #TODO: make a flag for this!
    self._model_fn = self.model_fn_builder(num_labels, self.hparams().learning_rate, FLAGS.train_steps)
    estimator = tf.estimator.Estimator(
        model_fn=self._model_fn,
        params=self.hparams(),
        config=tf.estimator.RunConfig(model_dir=model_dir))
    return estimator

  def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
    """Creates a classification model."""
    bert_module = hub.Module(
        FLAGS.model_spec,
        trainable=False)
    bert_inputs = dict(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)
    

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]
    tf.logging.log(tf.logging.INFO, '****output_layer shape: %s' % output_layer.get_shape())
    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    tf.logging.log(tf.logging.INFO, '****output_weights shape: %s' % output_weights.get_shape())
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())
    tf.logging.log(tf.logging.INFO, '****output_bias shape: %s' % output_bias.get_shape())

    with tf.variable_scope("loss"):
      # Dropout helps prevent overfitting
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

      logits = tf.matmul(output_layer, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      tf.logging.log(tf.logging.INFO, '****logits shape: %s' % logits.get_shape())
      log_probs = tf.nn.log_softmax(logits, axis=-1)

      # Convert labels into one-hot encoding
      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
      tf.logging.log(tf.logging.INFO, '****one_hot_labels shape: %s' % one_hot_labels.get_shape())
      predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
      tf.logging.log(tf.logging.INFO, '****predicted_labels shape: %s' % predicted_labels.get_shape())
      # If we're predicting, we want predicted labels and the probabiltiies.
      if is_predicting:
        return (predicted_labels, log_probs)

      # If we're train/eval, compute loss between predicted and actual label
      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      loss = tf.reduce_mean(per_example_loss)
      return (loss, predicted_labels, log_probs)

  def model_fn_builder(self, num_labels, learning_rate, num_train_steps,
                     num_warmup_steps = 0):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
      segment_ids = features["segment_ids"]
      label_ids = features["label_ids"]

      is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
      # TRAIN and EVAL
      if not is_predicting:

        (loss, predicted_labels, log_probs) = self.create_model(
          is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

        # TODO: make better optimizer and use learning_rate flag
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())

        # Calculate evaluation metrics
        def metric_fn(label_ids, predicted_labels):
          accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
          f1_score = tf.contrib.metrics.f1_score(
              label_ids,
              predicted_labels)
          auc = tf.metrics.auc(
              label_ids,
              predicted_labels)
          recall = tf.metrics.recall(
              label_ids,
              predicted_labels)
          precision = tf.metrics.precision(
              label_ids,
              predicted_labels) 
          true_pos = tf.metrics.true_positives(
              label_ids,
              predicted_labels)
          true_neg = tf.metrics.true_negatives(
              label_ids,
              predicted_labels)   
          false_pos = tf.metrics.false_positives(
              label_ids,
              predicted_labels)  
          false_neg = tf.metrics.false_negatives(
              label_ids,
              predicted_labels)
          return {
              "eval_accuracy": accuracy,
              "f1_score": f1_score,
              "auc": auc,
              "precision": precision,
              "recall": recall,
              "true_positives": true_pos,
              "true_negatives": true_neg,
              "false_positives": false_pos,
              "false_negatives": false_neg
          }

        eval_metrics = metric_fn(label_ids, predicted_labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode=mode,
              loss=loss,
              eval_metric_ops=eval_metrics)
      else:
        (predicted_labels, log_probs) = self.create_model(
          is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

        predictions = {
            'probabilities': log_probs,
            'labels': predicted_labels
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn
