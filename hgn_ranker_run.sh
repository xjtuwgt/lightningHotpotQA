#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/

LONG_FORMER_ROOT=allenai
ELECTRA_ROOT=google
SELECTEED_DOC_NUM=4


PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only

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
#    INPUTS=("hotpot_train_v1.1.json;train")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT


        echo "4. HGN re-ranker model (4)"
        # Input: $INPUT_FILE, doc_link_ner.json,  ner.json, long_para_ranking.json
        # Output: long_multihop_para.json
        python hgnrankerDataUtils/hgnmodel_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED/long_para_ranking.json $OUTPUT_PROCESSED/long_multihop_para.json $SELECTEED_DOC_NUM

#        echo "5. Dump features for reberta (5) do_lower_case"
#        python jdscripts/5_ext_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --do_lower_case --model_name_or_path roberta-large --ner_path $OUTPUT_PROCESSED/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker long --data_type $DATA_TYPE#
#
#        echo "5. Dump features for albert (5) do_lower_case"
#        python jdscripts/5_ext_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker long --data_type $DATA_TYPE

    done

}

for proc in "download" "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done
