#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# Re-rank the paragraphs according to the trained HGN models
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
    INPUTS=("hotpot_train_v1.1.json;train")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT

#        echo "1. Splitting 10 paras into (4, 4, 2)"
#        # Input: INPUT_FILE, train_long_para_ranking.json
#        # Output: split_train_long_para_ranking.json
#        python sr_mhqa/hotpotqa_rank_split.py --full_data $INPUT_FILE --rank_data $OUTPUT_PROCESSED/train_long_para_ranking.json --split_rank_data $OUTPUT_PROCESSED/split_train_long_para_ranking.json

#        echo "2. Positive/negative para preprocess, tokenize (albert)"
#        # Input: INPUT_FILE, split_train_long_para_ranking.json
#        # Output: Example dictionary
#        python sr_mhqa/hotpotqa_sr_dump_examples.py --full_data $INPUT_FILE --split_rank_data $OUTPUT_PROCESSED/split_train_long_para_ranking.json --model_name_or_path albert-xxlarge-v2 --do_lower_case --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE

        echo "4. Drop sentence testing (albert low)"
        python sr_mhqa/hotpotqa_sr_test_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE

    done
}

for proc in "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done
