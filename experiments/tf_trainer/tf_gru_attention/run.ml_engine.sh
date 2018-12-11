#!/bin/bash
# This script runs one training job on Cloud MLE.

# Note:
# We currently use 2 different embeddings:
# - glove.6B/glove.6B.300d.txt
# - google-news/GoogleNews-vectors-negative300.txt
# Glove assumes all words are lowercased, while Google-news handles different casing.
# As there is currently no tf operation that perform lowercasing, we have the following 
# requirements:
# - For google news: Run preprocess_in_tf=True (no lowercasing).
# - For glove.6B, Run preprocess_in_tf=False (will force lowercasing).

GCS_RESOURCES="gs://kaggle-model-experiments/resources"
DATETIME=`date '+%Y%m%d_%H%M%S'`
MODEL_NAME="tf_gru_attention"


if [ "$1" == "civil_comments" ]; then
    
    train_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/train-*.tfrecord"
    valid_path="${GCS_RESOURCES}/civil_comments_data/train_eval_test/eval-*.tfrecord"
    labels="toxicity"
    label_dtypes="float"
    train_steps=250000
    eval_period=800
    eval_steps=50
    learning_rate=0.00043202873559206826
    dropout_rate=0.45286591704272644
    gru_units="128"
    attention_units=64
    dense_units="128"
    batch_size=32

elif [ "$1" == "toxicity" ]; then

    train_path="${GCS_RESOURCES}/toxicity_q42017_train.tfrecord"
    valid_path="${GCS_RESOURCES}/toxicity_q42017_validate.tfrecord"
    labels="frac_neg"
    label_dtypes="float"
    train_steps=110000
    eval_period=800
    eval_steps=50
    learning_rate=0.00067158439070251973
    dropout_rate=0.0030912176559074966
    gru_units="128"
    attention_units=32
    dense_units="64"
    batch_size=64

elif [ "$1" == "many_communities" ]; then

    train_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_train.tfrecord"
    valid_path="${GCS_RESOURCES}/transfer_learning_data/many_communities/20181105_validate.tfrecord"
    labels="removed"
    label_dtypes="int"
    train_steps=10000000
    eval_period=800
    eval_steps=50
    learning_rate=0.00036793562485846982
    dropout_rate=0.60134625061009983
    gru_units="128"
    attention_units=32
    dense_units="64,64"
    batch_size=16

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
    --embeddings_path="${GCS_RESOURCES}/glove.6B/glove.6B.100d.txt" \
    --model_dir="${JOB_DIR}/model_dir" \
    --labels=$labels \
    --label_dtypes=$label_dtypes \
    --train_steps=$train_steps \
    --eval_period=$eval_period \
    --eval_steps=$eval_steps \
    --preprocess_in_tf=False \
    --learning_rate=$learning_rate \
    --dropout_rate=$dropout_rate \
    --gru_units=$gru_units \
    --attention_units=$attention_units \
    --dense_units=$dense_units \
    --batch_size=$batch_size
