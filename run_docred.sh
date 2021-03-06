#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/
ELECTRA_ROOT=google

# define precached BERT MODEL path

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/hgntransformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

# 0. Build Database from Wikipedia

preprocess() {
    INPUTS=("converted_docred_total.json;docred")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT

#        echo "0. Build Database from DocRed"
#        # Input: INPUT_FILE
#        # Output: enwiki_ner_docred.db
#        python docredscripts/0_build_db_docred.py --data_path $INPUT_FILE --save_path $DATA_ROOT/knowledge/enwiki_ner_docred.db

#        echo "1. Extract Wiki Link & NER from DB"
#        # Input: INPUT_FILE, enwiki_ner_docred.db
#        # Output: doc_link_ner.json
#        python docredscripts/1_extract_db_docred.py $INPUT_FILE $DATA_ROOT/knowledge/enwiki_ner_docred.db $OUTPUT_PROCESSED/doc_link_ner.json
#
#        echo "2. Extract NER for Question and Context"
#        # Input: doc_link_ner.json
#        # Output: ner.json
#        python docredscripts/2_extract_ner_docred.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json

#        echo "3. Dump features for roberta (do_lower_case)"
#        python jdscripts/5_ext_dump_features.py --para_path $OUTPUT_PROCESSED/docred_multihop_para.json --full_data $INPUT_FILE --model_name_or_path roberta-large --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker docred --data_type $DATA_TYPE
#
#        echo "3. Dump features for roberta (do_lower_case) (SAE graph)"
#        python jdscripts/5_ext_graph_dump_features.py --para_path $OUTPUT_PROCESSED/docred_multihop_para.json --full_data $INPUT_FILE --model_name_or_path roberta-large --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker docred --data_type $DATA_TYPE --sae_graph

#        echo "3. Dump features for albert (do_lower_case)"
#        python jdscripts/5_ext_dump_features.py --para_path $OUTPUT_PROCESSED/docred_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker docred --data_type $DATA_TYPE
#
#        echo "3. Dump features for albert (do_lower_case) (SAE graph)"
#        python jdscripts/5_ext_graph_dump_features.py --para_path $OUTPUT_PROCESSED/docred_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker docred --data_type $DATA_TYPE --sae_graph

        echo "3. Dump features for electra (do_lower_case)"
        python jdscripts/5_ext_dump_features.py --para_path $OUTPUT_PROCESSED/docred_multihop_para.json --full_data $INPUT_FILE --model_name_or_path $ELECTRA_ROOT/electra-large-discriminator --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type electra --tokenizer_name $ELECTRA_ROOT/electra-large-discriminator --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker docred --data_type $DATA_TYPE

        echo "3. Dump features for electra (do_lower_case) (SAE graph)"
        python jdscripts/5_ext_graph_dump_features.py --para_path $OUTPUT_PROCESSED/docred_multihop_para.json --full_data $INPUT_FILE --model_name_or_path $ELECTRA_ROOT/electra-large-discriminator --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type electra --tokenizer_name $ELECTRA_ROOT/electra-large-discriminator --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --ranker docred --data_type $DATA_TYPE --sae_graph

#        echo "4. Test dumped features"
        #python scripts/6_test_features.py --full_data $INPUT_FILE --input_dir $OUTPUT_FEAT --output_dir $OUTPUT_FEAT --model_type roberta --model_name_or_path roberta-large
    done
}

for proc in "preprocess"
do
        $proc
done

