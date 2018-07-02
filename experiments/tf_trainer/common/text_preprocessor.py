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
"""Text Preprocessor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import functools
import tensorflow as tf
from tf_trainer.common import types
from tf_trainer.common import base_model
from tf_trainer.common.token_embedding_index import LoadTokenIdxEmbeddings
from typing import Tuple, Dict, Optional, List, Callable


class TextPreprocessor():
  """Text Preprocessor TensorFlow Estimator Extension.

  Uses embedding indexes to create tensors that map tokens (provided by an
  abstract tokenizer funtion) to embeddings.

  NOTE: You might be wondering why we don't go straight from the word to the
  embedding. The (maybe incorrect) thought process is that the embedding portion
  can be made a part of the tensorflow graph whereas the word to index portion
  can not (since words have variable length). Future work may include fixing a
  max word length.
  """

  def __init__(self, embeddings_path: str) -> None:
    self._word_to_idx, self._embeddings_matrix, self._unknown_token = (
      LoadTokenIdxEmbeddings(embeddings_path))  # type: Tuple[Dict[str, int], np.ndarray, int]

  def tokenize_tensor_op(self, tokenizer: Callable[[str], List[str]]
                        ) -> Callable[[types.Tensor], types.Tensor]:
    """Tensor op that converts some text into an array of ints that correspond
    with this preprocessor's embedding.
    """

    def _tokenize_tensor_op(text: types.Tensor) -> types.Tensor:

      def _tokenize(b: bytes) -> np.ndarray:
        return np.asarray([
            self._word_to_idx.get(w, self._unknown_token)
            for w in tokenizer(b.decode('utf-8'))
        ])

      return tf.py_func(_tokenize, [text], tf.int64)

    return _tokenize_tensor_op

  def add_embedding_to_model(self, model: base_model.BaseModel,
                             text_feature_name: str) -> base_model.BaseModel:
    """Returns a new BaseModel with an embedding layer prepended.

    Args:
      model: An existing BaseModel instance.
      text_feature_name: The name of the feature containing text.
    """

    return model.map(
        functools.partial(self.create_estimator_with_embedding,
                          text_feature_name))

  def create_estimator_with_embedding(
      self, text_feature_name: str,
      estimator: tf.estimator.Estimator) -> tf.estimator.Estimator:
    """Takes an existing estimator and prepends the embedding layers to it.

    Args:
      estimator: A predefined Estimator that expects embeddings.
      text_feature_name: The name of the feature containing the text.

    Returns:
      TF Estimator with embedding ops added.
    """
    old_model_fn = estimator.model_fn
    old_config = estimator.config
    old_params = estimator.params

    def new_model_fn(features, labels, mode, params, config):
      """model_fn used in defining the new TF Estimator"""

      embeddings = self.word_embeddings()

      text_feature = features[text_feature_name]
      # Make sure all examples are length 300
      # TODO: Parameterize 300
      text_feature = tf.pad(text_feature, [[0, 0], [0, 300]])
      text_feature = text_feature[:, 0:300]
      word_embeddings = tf.nn.embedding_lookup(embeddings, text_feature)
      new_features = {text_feature_name: word_embeddings}

      # Fix dimensions to make Keras model output match label dims.
      labels = {k: tf.expand_dims(v, -1) for k, v in labels.items()}
      return old_model_fn(new_features, labels, mode=mode, config=config)

    return tf.estimator.Estimator(
        new_model_fn, config=old_config, params=old_params)

  def word_to_idx(self) -> Dict[str, int]:
    return self._word_to_idx

  def unknown_token(self) -> int:
    return self._unknown_token

  def word_to_idx_table(self) -> tf.contrib.lookup.HashTable:
    """Get word to index mapping as a TF HashTable."""

    keys = list(self._word_to_idx.keys())
    values = list(self._word_to_idx.values())
    table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values),
        self._unknown_token)
    return table

  def word_embeddings(self, trainable=True) -> tf.Variable:
    """Get word embedding TF Variable."""

    embeddings_shape = self._embeddings_matrix.shape
    initial_embeddings_matrix = tf.constant_initializer(self._embeddings_matrix)
    embeddings = tf.get_variable(
        name='word_embeddings',
        shape=embeddings_shape,
        initializer=initial_embeddings_matrix,
        trainable=trainable)
    return embeddings
