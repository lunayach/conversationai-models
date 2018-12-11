#!/bin/bash
# This script runs one training job on Cloud MLE.

GCS_RESOURCES="gs://kaggle-model-experiments/resources"
DATETIME=`date '+%Y%m%d_%H%M%S'`
MODEL_NAME="tf_hub_classifier"

if [ "$1" == "civil_comments" ]; then
    
    train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord"
    train_steps=500000
    eval_period=800
    eval_steps=50
    labels="toxicity"
    label_dtypes="float"
    learning_rate=0.0043168262804966252
    dropout_rate=0.65536985859319385
    dense_units="512,128,64"
    batch_size=16

elif [ "$1" == "toxicity" ]; then

    train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord"
    valid_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord"
    train_steps=110000
    eval_period=800
    eval_steps=50
    labels="frac_neg"
    label_dtypes="float"
    learning_rate=0.00012841041406549914
    dropout_rate=0.48078070594727657
    dense_units="128,64,64"
    batch_size=64

elif [ "$1" == "many_communities" ]; then

    train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_train.tfrecord"
    valid_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_validate.tfrecord"
    train_steps=10000000
    eval_period=800
    eval_steps=50
    labels="removed"
    label_dtypes="int"
    learning_rate=0.00037031438056183672
    dropout_rate=0.060691745638165928
    dense_units="128,128,128,64"
    batch_size=16

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    return;
fi


MODEL_NAME_DATA=${MODEL_NAME}_$1
JOB_DIR=gs://kaggle-model-experiments/tf_trainer_runs/${USER}/${MODEL_NAME_DATA}/${DATETIME}


gcloud ml-engine jobs submit training tf_trainer_${MODEL_NAME_DATA}_${USER}_${DATETIME} \
    --job-dir=${JOB_DIR} \
    --runtime-version=1.10 \
    --scale-tier 'BASIC_GPU' \
    --module-name="tf_trainer.${MODEL_NAME}.run" \
    --package-path=tf_trainer \
    --python-version "3.5" \
    --region=us-east1 \
    --verbosity=debug \
    -- \
    --train_path=$train_path \
    --validate_path=$valid_path \
    --model_dir="${JOB_DIR}/model_dir" \
    --labels=$labels \
    --label_dtypes=$label_dtypes \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps \
    --learning_rate=$learning_rate \
    --dropout_rate=$dropout_rate \
    --dense_units=$dense_units \
    --batch_size=$batch_size