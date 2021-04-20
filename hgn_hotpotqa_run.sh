#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/


PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only

# define precached BERT MODEL path
ROBERTA_LARGE=$DATA_ROOT/models/pretrained/roberta-large

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/hgntransformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

mkdir -p $DATA_ROOT/models/pretrained_cache

preprocess() {

     echo "HGN roberta 103 re-ranker model hgn topk = 2"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdhgn.json --topk_para_num 2
     echo "HGN re-ranker model hgn topk = 3"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdhgn.json --topk_para_num 3

     echo "HGN roberta 103 re-ranker model long topk = 2"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdlong.json --topk_para_num 2
     echo "HGN roberta 103 re-ranker model long topk = 3"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdlong.json --topk_para_num 3

     echo "HGN roberta 3901  re-ranker model long topk = 3"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdhgn.long.json --topk_para_num 3
     echo "HGN roberta 3901 re-ranker model long topk = 2"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdhgn.long.json --topk_para_num 2

     echo "HGN roberta 3901 roberta 103 re-ranker model hgn topk = 3"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdhgn.hgn.json --topk_para_num 3
     echo "HGN re-ranker model hgn topk = 2"
     python3 jd_para_reranker.py --config_file configs/predict.roberta.jdhgn.hgn.json --topk_para_num 2

     echo "HGN albert re-ranker model hgn topk = 2"
     python3 jd_albert_para_reranker.py --config_file configs/predict.albert.hgn.orig.json --topk_para_num 2
     echo "HGN albert re-ranker model hgn topk = 3"
     python3 jd_albert_para_reranker.py --config_file configs/predict.albert.hgn.orig.json --topk_para_num 3

     echo "HGN albert re-ranker model long topk = 2"
     python3 jd_albert_para_reranker.py --config_file configs/predict.albert.long.orig.json --topk_para_num 2
     echo "HGN albert re-ranker model long topk = 3"
     python3 jd_albert_para_reranker.py --config_file configs/predict.albert.long.orig.json --topk_para_num 3
}

for proc in "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done
