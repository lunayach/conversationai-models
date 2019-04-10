"""Experiments with Toxicity Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_trainer.common import base_model
from tf_trainer.common import model_trainer
from tf_trainer.common import serving_input
from tf_trainer.common import tfrecord_input
from tf_trainer.bert_classifier import model as bert_classifier

import tensorflow as tf
import tensorflow_hub as hub

FLAGS = tf.app.flags.FLAGS


def main(argv):
  del argv  # unused
  dataset = tfrecord_input.BERTReader()
  model = bert_classifier.BERTClassifierModel()

  trainer = model_trainer.ModelTrainer(dataset, model)
  trainer.train_with_eval()

  #serving_input_fn = serving_input.create_text_serving_input_fn(
  #    text_feature_name=base_model.TEXT_FEATURE_KEY,
  #    example_key_name=base_model.EXAMPLE_KEY)
  #trainer.export(serving_input_fn, base_model.EXAMPLE_KEY,
  #  metrics_key="auc/%s" % FLAGS.labels.split(',')[0])


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
