#!/bin/bash

source "tf_trainer/common/bert_dataset_config.sh"

python -m tf_trainer.bert_classifier.run \
  --train_path=$train_path \
  --validate_path=$valid_path \
  --model_dir="bert_classifier_local_model_dir" \
  --model_spec="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1" \
  --train_steps=100 \
  --eval_period=10
