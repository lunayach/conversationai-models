#!/bin/bash

GCS_RESOURCES="gs://kaggle-model-experiments/resources"
DATETIME=`date '+%Y%m%d_%H%M%S'`
MODEL_NAME="tf_cnn"

if [ "$1" == "civil_comments" ]; then
    
    train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord"
    train_steps=50000
    eval_period=800
    eval_steps=50
    labels="toxicity"
    label_dtypes="float"
    batch_size=128
    learning_rate=5.8524404808865205e-05
    dropout_rate=0.94922247211093835
    filter_sizes="5,5"
    num_filters=128
    dense_units="64,64"
    pooling_type="max"

elif [ "$1" == "toxicity" ]; then

    train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord"
    valid_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord"
    train_steps=50000
    eval_period=800
    eval_steps=50
    labels="frac_neg"
    label_dtypes="float"
    batch_size=128
    learning_rate=0.0003070178448483886
    dropout_rate=0.24996995568218527
    filter_sizes="5,5"
    num_filters=128
    dense_units="64,64"
    pooling_type="max"

elif [ "$1" == "many_communities" ]; then

    train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_train.tfrecord"
    valid_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_validate.tfrecord"
    train_steps=2000000
    eval_period=800
    eval_steps=50
    labels="removed"
    label_dtypes="int"
    batch_size=64
    learning_rate=3.7980233501470653e-05
    dropout_rate=0.36109472326413428
    filter_sizes="3,4,5"
    num_filters=128
    dense_units="128,128"
    pooling_type="max"

else
    echo "First positional arg must be one of civil_comments, toxicity, many_communities."
    return;
fi


MODEL_NAME_DATA=${MODEL_NAME}_$1_glove
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
    --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.300d.txt" \
    --model_dir="${JOB_DIR}/model_dir" \
    --is_embedding_trainable=False \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps \
    --labels=$labels \
    --label_dtypes=$label_dtypes \
    --preprocess_in_tf=False \
    --batch_size=$batch_size \
    --learning_rate=$learning_rate \
    --dropout_rate=$dropout_rate \
    --filter_sizes=$filter_sizes \
    --num_filters=$num_filters \
    --dense_units=$dense_units \
    --pooling_type=$pooling_type

echo "Model dir:"
echo ${JOB_DIR}/model_dir