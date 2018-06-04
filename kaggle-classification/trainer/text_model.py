import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
#import os
import pandas as pd
#import re
#import seaborn as sns

def main(FLAGS):

  """# Getting started

  ## Data
  We will try to solve the [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/) task from Mass et al. The dataset consists of IMDB movie reviews labeled by positivity from 1 to 10. The task is to label the reviews as **negative** or **positive**.
  """

  print('in main!!!!')

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

  # Download and process the dataset files.
  # TODO: this doesn't seem to work on ml-engine
  def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df

  # Reduce logging output.
  tf.logging.set_verbosity(tf.logging.ERROR)

  # TODO: remove hardcoded data
  train = {
    'sentence': ['this is something good', 'and this is something bad'],
    'polarity': [1, 0],
  }
  test = {
    'sentence': ['here is something else good', 'here is something else bad'],
    'polarity': [1, 0],
  }

  train_df, test_df = pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test)

  """## Model
  ### Input functions

  [Estimator framework](https://www.tensorflow.org/get_started/premade_estimators#overview_of_programming_with_estimators) provides [input functions](https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/pandas_input_fn) that wrap Pandas dataframes.
  """

  # Training input on the whole training set with no limit on training epochs.
  train_input_fn = tf.estimator.inputs.pandas_input_fn(
      train_df, train_df["polarity"], num_epochs=None, shuffle=True)

  # Prediction on the whole training set.
  predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
      train_df, train_df["polarity"], shuffle=False)
  # Prediction on the test set.
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

  """### Estimator

  For classification we can use a [DNN Classifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) (note further remarks about different modelling of the label function at the end of the tutorial).
  """

  estimator = tf.estimator.DNNClassifier(
      hidden_units=[500, 100],
      feature_columns=[embedded_text_feature_column],
      n_classes=2,
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

  """### Training

  Train the estimator for a reasonable amount of steps.
  """

  # Training for 1,000 steps means 128,000 training examples with the default
  # batch size. This is roughly equivalent to 5 epochs since the training dataset
  # contains 25,000 examples.
  estimator.train(input_fn=train_input_fn, steps=1000);

  """# Prediction

  Run predictions for both training and test set.
  """

  train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
  test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

  print("Training set accuracy: {accuracy}".format(**train_eval_result))
  print("Test set accuracy: {accuracy}".format(**test_eval_result))

  feature_spec = {
    'sentence': tf.FixedLenFeature(dtype=tf.string, shape=100)
  }
  serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

  print('about to call export_savedmodel')
  return_value = estimator.export_savedmodel(FLAGS.job_dir, serving_input_fn)
  print('export_savedmodel returned %s' % return_value)


  # your_feature_spec = {
  #     "sentence": tf.FixedLenFeature(dtype=tf.string, shape=1)
  # }

  # def _serving_input_receiver_fn():
  #     serialized_tf_example = tf.placeholder(dtype=tf.string, shape=None,
  #                                            name='input_example_tensor')
  #     # key (e.g. 'examples') should be same with the inputKey when you
  #     # buid the request for prediction
  #     receiver_tensors = {'examples': serialized_tf_example}
  #     features = tf.parse_example(serialized_tf_example, your_feature_spec)
  #     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  # estimator.export_savedmodel('gs://kaggle-model-experiments/dborkan', _serving_input_receiver_fn)

# end of main

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--job-dir", type=str, default="", help="The directory where the job is staged")
  FLAGS = parser.parse_args()

  main(FLAGS)
