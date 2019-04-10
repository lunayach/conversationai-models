#!/bin/bash

BASE_PATH="gs://conversationai-models"
GCS_RESOURCES="${BASE_PATH}/resources"
MODEL_PARENT_DIR="${BASE_PATH}/tf_trainer_runs"

if [ "$1" == "civil_comments" ]; then
    train_path="${GCS_RESOURCES}/civil_comments_data/bert_train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/bert_train_eval_test/eval-*.tfrecord"

else
    echo "First positional arg must civil_comments."
    exit 1
fi
