import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
#import os
import pandas as pd
#import re

# Shows how to create a TF Estimator that takes in an unparsed string
# (as opposed to a vector or ints, etc), using tf.text_embedding_column
# based on tutorial at https://www.tensorflow.org/tutorials/text_classification_with_tf_hub

def main(FLAGS):
  # Get data.  train_df and test_df will both be Pandas dataframes containing the columns:
  #  'sentence': a variable length string, e.g. 'This movie was terrible!'
  #  'polarity': an int, 0 for negative reviews, 1 for positive reviews
  # Any other columns returned should be ignored
  train_df, test_df = get_hardcoded_data()

  # Estimator input functions
  train_input_fn = tf.estimator.inputs.pandas_input_fn(
      train_df, train_df["polarity"], num_epochs=None, shuffle=True)
  predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
      train_df, train_df["polarity"], shuffle=False)
  predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
      test_df, test_df["polarity"], shuffle=False)

  """### Feature columns

  TF-Hub provides a [feature column](https://github.com/tensorflow/hub/blob/master/docs/api_docs/python/hub/text_embedding_column.md) that applies a module on the given text feature and passes further the outputs of the module. In this tutorial we will be using the [nnlm-en-dim128 module](https://tfhub.dev/google/nnlm-en-dim128/1). For the purpose of this tutorial, the most important facts are:

  * The module takes **a batch of sentences in a 1-D tensor of strings** as input.
  * The module is responsible for **preprocessing of sentences** (e.g. removal of punctuation and splitting on spaces).
  * The module works with any input (e.g. **nnlm-en-dim128** hashes words not present in vocabulary into ~20.000 buckets).
  """
  embedded_text_feature_column = hub.text_embedding_column(
      key="sentence",
      module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
  # Create and train the estimator, parameters are copied from the tutorial
  estimator = tf.estimator.DNNClassifier(
      hidden_units=[500, 100],
      feature_columns=[embedded_text_feature_column],
      n_classes=2,
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))
  estimator.train(input_fn=train_input_fn, steps=1000);

  # Evaluate the model
  train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
  print("Training set accuracy: {accuracy}".format(**train_eval_result))
  test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)
  print("Test set accuracy: {accuracy}".format(**test_eval_result))

  # Export a saved model
  feature_spec = {
    # TODO: what is shape?  why can't i use var length?
    'sentence': tf.FixedLenFeature(dtype=tf.string, shape=1)
  }
  serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
  saved_model_dir = estimator.export_savedmodel(FLAGS.job_dir, serving_input_fn)
  print('export_savedmodel wrote to %s' % saved_model_dir)

# end of main



# Download and process the dataset files.
# TODO: use this!
def download_and_load_datasets(force_download=False):
  # Load all files from a directory in a DataFrame.
  def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
      with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
        data["sentence"].append(f.read())
        data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

  # Merge positive and negative examples, add a polarity column and shuffle.
  def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz",
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
      extract=True)
  train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      "aclImdb", "test"))
  return train_df, test_df


def get_hardcoded_data():
  train = {
    'sentence': ['this is something good', 'and this is something bad'],
    'polarity': [1, 0],
  }
  test = {
    'sentence': ['here is something else good', 'here is something else bad'],
    'polarity': [1, 0],
  }
  return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--job-dir", type=str, default="", help="The directory where the job is staged")
  FLAGS = parser.parse_args()
  main(FLAGS)
