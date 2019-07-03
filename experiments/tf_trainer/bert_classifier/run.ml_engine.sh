#!/bin/bash
# This script runs one training job on Cloud MLE.

source "tf_trainer/common/bert_dataset_config.sh"
DATETIME=$(date '+%Y%m%d_%H%M%S')
MODEL_NAME="bert_classifier"
MODEL_NAME_DATA="${MODEL_NAME}_$1"
JOB_DIR="${MODEL_PARENT_DIR}/${USER}/${MODEL_NAME_DATA}/${DATETIME}"


if [ "$1" == "civil_comments" ]; then
    batch_size=128
    dropout_rate=0.12298246947263007
    learning_rate=0.0001473127671008433
    train_steps=50000
    eval_period=1000
    eval_steps=2000
    config="tf_trainer/common/p100_config.yaml"

else
    echo "First positional arg must be civil_comments."
    return;
fi


gcloud ml-engine jobs submit training tf_trainer_${MODEL_NAME_DATA}_${USER}_${DATETIME} \
    --job-dir=${JOB_DIR} \
    --runtime-version=1.12 \
    --config $config \
    --module-name="tf_trainer.${MODEL_NAME}.run" \
    --package-path=tf_trainer \
    --region=us-east1 \
    --verbosity=debug \
    -- \
    --train_path=$train_path \
    --validate_path=$valid_path \
    --model_dir="${JOB_DIR}/model_dir" \
    --batch_size=$batch_size \
    --dropout_rate=$dropout_rate \
    --learning_rate=$learning_rate \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps \
    --model_spec="gs://conversationai-models/resources/tfhub/bert_uncased_L-12_H-768_A-12-1/5a395eafef2a37bd9fc55d7f6ae676d2a134a838" \
    --n_export=1

