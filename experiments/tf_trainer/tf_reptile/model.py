"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'filter_sizes', '5',
    'Comma delimited string for the sizes of convolution filters.')
tf.app.flags.DEFINE_integer(
    'num_filters', 128,
    'Number of convolutional filters for every convolutional layer.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
    'dense_units', '128',
    'Comma delimited string for the number of hidden units in the dense layer.')
tf.app.flags.DEFINE_integer('embedding_size', 300,
                            'The number of dimensions in the word embedding.')
tf.app.flags.DEFINE_string('pooling_type', 'average', 'Average or max pooling.')


DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)


class TextCNNModel:
  """
  A CNN model for text classification.
  """
  def __init__(self, num_classes, max_seq_length, embedding_dim,
               optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
    self.filter_sizes = [int(units) for units in FLAGS.filter_sizes.split(',')]
    self.dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    self.input_ph = tf.placeholder(tf.float32,
        shape=(None, max_seq_length, embedding_dim))
    
    X = self.input_ph

    # Conv
    for filter_size in self.filter_sizes:
      X = layers.Conv1D(
          FLAGS.num_filters,
          filter_size,
          activation='relu',
          padding='same')(X)
    if FLAGS.pooling_type == 'average':
      X = layers.GlobalAveragePooling1D()(X)
    elif FLAGS.pooling_type == 'max':
      X = layers.GlobalMaxPooling1D()(X)
    else:
      raise ValueError('Unrecognized pooling type parameter')

    # FC
    logits = X
    for num_units in self.dense_units:
      logits = tf.layers.dense(
          inputs=logits, units=num_units, activation=tf.nn.relu)
      logits = tf.layers.dropout(logits, rate=FLAGS.dropout_rate)
    self.logits = tf.layers.dense(
        inputs=logits, units=num_classes, activation=None)
    
    self.label_ph = tf.placeholder(tf.int32, shape=(None,))
    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                               logits=self.logits)
    self.predictions = tf.argmax(self.logits, axis=-1)
    self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)