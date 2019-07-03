#!/bin/bash

FILENAMES='eval-00001-of-00003.tfrecord,'\
'eval-00002-of-00003.tfrecord,'\
'test-00000-of-00004.tfrecord,'\
'test-00001-of-00004.tfrecord,'\
'test-00002-of-00004.tfrecord,'\
'test-00003-of-00004.tfrecord,'\
'train-00000-of-00005.tfrecord,'\
'train-00002-of-00005.tfrecord,'\
'train-00003-of-00005.tfrecord,'\
'train-00004-of-00005.tfrecord'

INPUT_DATA_PATH='gs://conversationai-models/resources/civil_comments_data/train_eval_test/'
OUTPUT_DATA_PATH='gs://conversationai-models/resources/civil_comments_data/bert_train_eval_test/'
BERT_URL='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
MAX_SEQ_LENGTH=256

echo """
Running...
python bert_tfrecord_converter.py \
 --filenames=$FILENAMES \
 --input_data_path=$INPUT_DATA_PATH \
 --output_data_path=$OUTPUT_DATA_PATH \
 --bert_url=$BERT_URL \
 --max_sequence_length=$MAX_SEQ_LENGTH
"""

python bert_tfrecord_converter.py \
 --filenames=$FILENAMES \
 --input_data_path=$INPUT_DATA_PATH \
 --output_data_path=$OUTPUT_DATA_PATH \
 --bert_url=$BERT_URL \
 --max_sequence_length=$MAX_SEQ_LENGTH