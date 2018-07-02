"""Keras CNN Model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tf_trainer.common import base_keras_model
from tf_trainer.common import types
from tf_trainer.common import cnn_spec_parser
import tensorflow as tf

from typing import Set, List

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
# TODO: Add validation
tf.app.flags.DEFINE_integer(
  'max_input_seq_length', 300,
  'The max length (to truncate to) of an input example in tokens.')
tf.app.flags.DEFINE_integer(
  'word_embedding_size', 100,
  'The size of the input word embeddings. Should match the embeddings used.')
tf.app.flags.DEFINE_float(
  'learning_rate', 0.001,
  'The learning rate to use during training.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_float(
  'dropout_fraction', 0.1,
  'The fraction of inputs drop during training.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
  'cnn_shape',
  '(2/2 -> 100), (3/2 -> 100) : (6/2 -> 100) : (3/1 -> 100)',
  'The shape of the convolutional layers. '
  'See: tf_trainer_common/cnn_spec_parser.py')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
  'dense_units', '1024',
  'Comma delimited string for the number of hidden units in the dense layer.')


class KerasCNNModel(base_keras_model.BaseKerasModel):
  """Keras CNN Model

  Keras implementation of a CNN for text classification. Inputs should be
  sequences of word embeddings.
  """

  def __init__(self, labels: Set[str], optimizer='adam') -> None:
    self.cnn_layers_spec = cnn_spec_parser.SequentialLayers(FLAGS.cnn_shape)
    self._labels = labels

  def hparams(self):
    dense_units = [int(units) for units in FLAGS.dense_units.split(',')]
    return tf.contrib.training.HParams(
        learning_rate=float(FLAGS.learning_rate),
        dropout_fraction=float(FLAGS.dropout_fraction),
        cnn_shape=FLAGS.cnn_shape,
        dense_units=dense_units)

  # Local function you are expected to overwrite.
  def _get_keras_model(self) -> models.Model:
    I = layers.Input(
        shape=(FLAGS.max_input_seq_length, FLAGS.word_embedding_size),
        dtype='float32',
        name='comment_text')

    X = I # type: types.Tensor
    concurrent_filters = [X]
    for l in self.cnn_layers_spec.layers:
      X = layers.concatenate(concurrent_filters)
      concurrent_filters = []
      for f in l.filters: # type: cnn_spec_parser.Filter
        conv_filter = layers.Conv1D(f.num_filters, f.size, f.stride,
            activation='relu', padding='same')(X)
        concurrent_filters.append(conv_filter)

    # Max pooling for each final filter layer to get to a fixed output size.
    X = layers.concatenate([layers.GlobalAveragePooling1D()(c) for c in concurrent_filters])

    # for conv_filter in concurrent_filters:
    #   X = layers.GlobalAveragePooling1D()(X)
    # # Concurrent convolutional Layers of different sizes
    # conv_pools = [] # type: List[types.Tensor]
    # for filter_size in self.hparams().filter_sizes:
    #     x_conv = layers.Conv1D(self.hparams().num_filters, filter_size,
    #         activation='relu', padding='same')(I)
    #     x_pool = layers.GlobalAveragePooling1D()(x_conv)
    #     conv_pools.append(x_pool)
    # X = layers.concatenate(conv_pools) # type: types.Tensor

    # Dense Layers after convolutions
    for num_units in self.hparams().dense_units:
      X = layers.Dense(num_units, activation='relu')(X)
      X = layers.Dropout(self.hparams().dropout_fraction)(X)

    # Outputs
    outputs = []
    for label in self._labels:
      outputs.append(layers.Dense(1, activation='sigmoid', name=label)(X))

    model = models.Model(inputs=I, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(lr=self.hparams().learning_rate),
        loss='binary_crossentropy',
        metrics=['binary_accuracy', super().roc_auc])

    tf.logging.info(model.summary())
    return model
