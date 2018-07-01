# Text Classification Framework

This directory contains an ML framework for text classification. We illustrate
it with toxic (and other attributes) comment classification.

The framework is structured as a series of common files and templates to quickly
construct models on top of the [Keras](https://keras.io/) or the [TensorFlow
Estimator API](https://www.tensorflow.org/programmers_guide/estimators).

The templates also demonstrate how these models can be trained using [Google ML
Engine](https://cloud.google.com/ml-engine/) and track experiments with
[Comet.ML](https://www.comet.ml/).


## Environment Setup

### Build Tools/Bazel Dependencies

Install [Bazel](https://docs.bazel.build/versions/master/install-os-x.html);
this is the build tool we use to run tests, etc.

### Python Dependencies and Setup

Install library dependencies (it is optional, but recommended to install these
in a [Virtual Environment](https://docs.python.org/3/tutorial/venv.html):

    ```shell
    # The python3 way to create and use virtual environment
    # (optional, but recommended):
    python3 -m venv .pyenv
    source .pyenv/bin/activate
    # Install dependencies
    pip install -r requirements.txt

    # Create the config template
    cp config.template.py config.py
    # ... now set the relevant vars your 'config.py'; see below ...

    # ... do stuff ...

    # Exist your virtual environment.
    deactivate
    ```

### Cloud and ML Engine configuration

You must set your Google cloud project config to the project that you will be
training on, and which has read/write access to the cloud storage bucket you are
using to store data and results.

```shell
# Set your project
gcloud config set project `<you project name>`
# Authenticate with Google Cloud
gcloud auth login
```

Your project must also have enabled the Google Cloud ML Engine APIs, as well as
Google Cloud Storage.

### Comet.ML configuration

If you have a [Comet ML](https://www.comet.ml/) key, you can use that platform
to track experiments and monitor your model training progress and quality.
Simply add your api key in the `config.py` file.

## Training an Existing Model

To train an existing model, execute either command:
 * `./tf_trainer/MODEL_NAME/run.local.sh` to run training locally, or
 * `./tf_trainer/MODEL_NAME/run.ml_engine.sh` to run training on [Google ML
Engine](https://cloud.google.com/ml-engine/).

These scripts assume that you have access to the resources on our cloud
projects. If you don't, you can still run the models locally, but will have to
modify the data paths in `run.local.sh`. At the moment, we only support reading
data in `tf.record` format. See
[`tools/convert_csv_to_tfrecord.py`](https://github.com/conversationai/conversationai-models/blob/master/experiments/tools/convert_csv_to_tfrecord.py)
for a simple CSV to `tf.record` converter.

Once training has been started, you can look at the status of training with
Tensorboard on your local machine:

```shell
tensorboard --logdir='<path to logs directory; can be GCS path>'
```

## Evaluate an Existing Model on New Data

TODO(nthain)


## Development

### Type Checking

Check the typings:

```shell
mypy --ignore-missing-imports -p tf_trainer
```

It's recommended you use mypy as an additional linter in your editor.

### Testing

Run all the tests and see the output streamed:

```shell
bazel test --test_output=streamed ...
```

You can also run tests individually, directly with python like so:

```shell
python -m tf_trainer.common.tfrecord_input_test
python -m tf_trainer.common.base_keras_model_test
```

### Building a New Model

TODO(jjtan)
