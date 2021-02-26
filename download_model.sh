#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/hgntransformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache


# 0. Download Roberta pretrained model: ahotrod/roberta_large_squad2
ROBERTA_SQUAD2=https://huggingface.co/ahotrod/roberta_large_squad2/resolve/main
roberta() {
    [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/models/pretrained/ahotrod/roberta_large_squad2

    wget -P $DATA_ROOT/models/pretrained/ahotrod/roberta_large_squad2 $ROBERTA_SQUAD2/config.json
    wget -P $DATA_ROOT/models/pretrained/ahotrod/roberta_large_squad2 $ROBERTA_SQUAD2/vocab.json
    wget -P $DATA_ROOT/models/pretrained/ahotrod/roberta_large_squad2 $ROBERTA_SQUAD2/tokenizer_config.json
    wget -P $DATA_ROOT/models/pretrained/ahotrod/roberta_large_squad2 $ROBERTA_SQUAD2/pytorch_model.bin
}

ALBERT_SQUAD2=https://huggingface.co/mfeb/albert-xxlarge-v2-squad2/resolve/main
albert() {
  [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/models/pretrained/mfeb/albert-xxlarge-v2-squad2

  wget -P $DATA_ROOT/models/pretrained/mfeb/albert-xxlarge-v2-squad2 $ALBERT_SQUAD2/config.json
  wget -P $DATA_ROOT/models/pretrained/mfeb/albert-xxlarge-v2-squad2 $ALBERT_SQUAD2/tokenizer_config.json
  wget -P $DATA_ROOT/models/pretrained/mfeb/albert-xxlarge-v2-squad2 $ALBERT_SQUAD2/pytorch_model.bin
  wget -P $DATA_ROOT/models/pretrained/mfeb/albert-xxlarge-v2-squad2 $ALBERT_SQUAD2/special_tokens_map.json
}

for proc in "roberta" "albert"
do
    $proc
done