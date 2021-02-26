#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/hgntransformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache


# 0. Download Roberta pretrained model: ahotrod/roberta_large_squad2
ROBERTA_SQUAD2=https://huggingface.co/ahotrod/roberta_large_squad2/resolve/main/
roberta() {
    [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/models/pretrained/roberta-large/ahotrod

    wget -P $DATA_ROOT/models/pretrained/roberta-large/ahotrod $ROBERTA_SQUAD2/config.json
}

for proc in "roberta"
do
    $proc
done
