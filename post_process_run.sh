#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/

LONG_FORMER_ROOT=allenai
SELECTEED_DOC_NUM=6
TOPK_PARA_NUM=4


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
#    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor" "hotpot_train_v1.1.json;train")
    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor")
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

#        echo "1. Extract Wiki Link & NER from DB"
#        # Input: INPUT_FILE, enwiki_ner.db
#        # Output: doc_link_ner.json
#        python leaderboardscripts/1_lb_extract_db.py $INPUT_FILE $DATA_ROOT/knowledge/enwiki_ner.db $OUTPUT_PROCESSED/doc_link_ner.json
#
#        echo "2. Extract NER for Question and Context"
#        # Input: doc_link_ner.json
#        # Output: ner.json
#        python leaderboardscripts/2_lb_extract_ner.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json
#
#        echo "3. Paragraph ranking (1): longformer retrieval data preprocess"
#        # Output: para_ir_combined.json
#        python leaderboardscripts/3_lb_longformer_dataprepare_para_sel.py $INPUT_FILE $OUTPUT_PROCESSED/para_ir_combined.json
#
#        echo "3. Paragraph ranking (2): longformer retrieval ranking scores"
#        # switch to Longformer for final leaderboard, PYTORCH LIGHTING + '1.0.8' TRANSFORMER (3.3.1)
#        # Output: long_para_ranking.json
#        python leaderboardscripts/3_lb_longformer_paragraph_ranking.py --data_dir $OUTPUT_PROCESSED --eval_ckpt $DATA_ROOT/models/finetuned/PS/longformer_pytorchlighting_model.ckpt --raw_data $INPUT_FILE --input_data $OUTPUT_PROCESSED/para_ir_combined.json
#       The above scripts are shared #########

#        echo "3. MultiHop Paragraph Selection (3)"
#        # Input: $INPUT_FILE, doc_link_ner.json,  ner.json, long_para_ranking.json
#        # Output: long_multihop_para.json
#        python leaderboardscripts/3_lb_longformer_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED/long_para_ranking.json $OUTPUT_PROCESSED/long_multihop_para.json $SELECTEED_DOC_NUM

#        echo "4. Dump features for albert do_lower_case"
#        # Input: $INPUT_FILE, long_multihop_para.json; model_type, model_name, doc_link_ner.json, ner.json
#        python post_feature_collection/5_lb_hotpotqa_dump_features.py  --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker long --data_type $DATA_TYPE --max_para_num $SELECTEED_DOC_NUM --topk $TOPK_PARA_NUM

#        echo "5. Re-rank over top k via the trained model"
#        python post_feature_collection/topk_rerankering.py --daug_type long_low --devf_type long_low --data_type $DATA_TYPE --max_para_num $SELECTEED_DOC_NUM --topk_para_num $TOPK_PARA_NUM

#        echo "6. Re-rank over top k and hyper-link "
#        # output: topk_long_multihop_para.json
#        python leaderboardscripts/5_lb_topk_hgnmodel_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED $SELECTEED_DOC_NUM $TOPK_PARA_NUM long_low

        echo "7. Dump features for albert do_lower_case for re-rank results"
        python post_feature_collection/5_lb_hotpotqa_dump_features.py --para_path $OUTPUT_PROCESSED --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker long --data_type $DATA_TYPE --do_rerank --topk $TOPK_PARA_NUM --max_para_num $SELECTEED_DOC_NUM

#        echo "8. Model prediction with albert"
#        python post_feature_collection/postprocess_lb_predict.py --daug_type long_low --devf_type long_low --data_type $DATA_TYPE --max_para_num $SELECTEED_DOC_NUM --topk_para_num $TOPK_PARA_NUM --do_rerank

    done

}

for proc in "download" "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done