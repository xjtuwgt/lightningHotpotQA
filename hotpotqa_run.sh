#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/


LONG_FORMER_ROOT=allenai
ELECTRA_ROOT=google
SELECTEED_DOC_NUM=4


# PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only

# define precached BERT MODEL path
# ROBERTA_LARGE=$DATA_ROOT/models/pretrained/roberta-large
# pip install -U spacy
# python -m spacy download en_core_web_lg

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/hgntransformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

mkdir -p $DATA_ROOT/models/pretrained_cache

preprocess() {
    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor" "hotpot_train_v1.1.json;train")
#    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT

#        echo "1. MultiHop Paragraph Selection (long)"
#        # Input: $INPUT_FILE, doc_link_ner.json,  ner.json, long_para_ranking.json
#        # Output: long_multihop_para.json
#        python longformerscripts/4_longformer_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED/long_para_ranking.json $OUTPUT_PROCESSED/long_multihop_para.json $SELECTEED_DOC_NUM

#        echo "2. Tokenized example extraction (Roberta low)"
#        python HotpotQAModel/hotpotqa_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path roberta-large --do_lower_case --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE
#
#        echo "3. Tokenized example extraction (Albert low)"
#        python HotpotQAModel/hotpotqa_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE

#        echo "4. Drop sentence testing (Roberta low)"
#        python HotpotQAModel/hotpotqa_test_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path roberta-large --do_lower_case --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE

        echo "4. Drop sentence testing (albert low)"
        python HotpotQAModel/hotpotqa_test_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT  --ranker long --data_type $DATA_TYPE


        done
}

for proc in  "preprocess"
do
    $proc
done
