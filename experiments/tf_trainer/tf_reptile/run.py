"""Experiments with toxicity, civil_comments, many_communities datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
import tensorflow as tf
import random

from tf_trainer.common import base_model
from tf_trainer.common import model_trainer
from tf_trainer.common import serving_input
from tf_trainer.common import text_preprocessor
from tf_trainer.common import tfrecord_input
from tf_trainer.common import types
from tf_trainer.tf_reptile import model
from tf_trainer.tf_reptile import reptile
from tf_trainer.tf_reptile import reptile_trainer
from tf_trainer.common import episodic_tfrecord_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('embeddings_path',
                           'local_data/glove.6B/glove.6B.100d.txt',
                           'Path to the embeddings file.')

tf.app.flags.DEFINE_integer('max_seq_len', 10000, 'Maximum sequence length.')
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          'The learning rate to use during training.')
tf.app.flags.DEFINE_float('dropout_rate', 0.3,
                          'The dropout rate to use during training.')
# This would normally just be a multi_integer, but we use string due to
# constraints with ML Engine hyperparameter tuning.
# TODO: add link to relevant public issue/bug/documentation?
tf.app.flags.DEFINE_string(
    'dense_units', '128',
    'Comma delimited string for the number of hidden units in the dense layer.')
tf.app.flags.DEFINE_integer('embedding_size', 300,
                            'The number of dimensions in the word embedding.')
tf.app.flags.DEFINE_boolean('pretrained', False, 'Evaluate a pre-trained model?')
tf.app.flags.DEFINE_integer('seed', 0, 'random seed')
tf.app.flags.DEFINE_string('checkpoint', 'model_checkpoint', 'Checkpoint directory.')
tf.app.flags.DEFINE_integer('classes', 5, 'Number of classes per inner task.')
tf.app.flags.DEFINE_integer('shots', 5, 'Number of examples per class.')
tf.app.flags.DEFINE_integer('train_shots', 0, 'Shots in a training batch.')
tf.app.flags.DEFINE_integer('inner_batch', 5, 'Inner batch size.')
tf.app.flags.DEFINE_integer('inner_iters', 20, 'Inner iterations.')
tf.app.flags.DEFINE_boolean('replacement', True, 'Sample with replacement?')
tf.app.flags.DEFINE_float('meta_step', 0.1, 'Meta-training step size.')
tf.app.flags.DEFINE_float('meta_step_final', 0.1, 'Meta-training step size by the end.')
tf.app.flags.DEFINE_integer('meta_batch', 1, 'Meta-training batch size.')
tf.app.flags.DEFINE_integer('meta_iters', 400000, 'Meta-training iterations.')
tf.app.flags.DEFINE_integer('eval_batch', 5, 'Eval inner batch size.')
tf.app.flags.DEFINE_integer('eval_iters', 50, 'Eval inner iterations.')
tf.app.flags.DEFINE_integer('eval_samples', 10000, 'Evaluation samples.')
tf.app.flags.DEFINE_integer('eval_interval', 10, 'Train steps per eval.')
tf.app.flags.DEFINE_float('weight_decay', 1.0, 'Weight decay rate.')
tf.app.flags.DEFINE_boolean('transductive', False, 'Evaluate all samples at once.')


def main(argv):
  del argv  # unused

  random.seed(FLAGS.seed)

  embeddings_path = FLAGS.embeddings_path

  preprocessor = text_preprocessor.TextPreprocessor(embeddings_path)

  nltk.download('punkt')
  train_preprocess_fn = preprocessor.train_preprocess_fn(nltk.word_tokenize)
  dataset = episodic_tfrecord_input.EpisodicTFRecordInput(FLAGS.train_path,
    FLAGS.dev_path)

  with tf.Session() as sess:


  text_model = model.TextModel()
  reptile = reptile.TFReptileModel()

  trainer = reptile_trainer.ReptileTrainer(dataset, sess)
  trainer.train()

  


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
