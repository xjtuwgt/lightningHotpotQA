#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/


preprocess() {
    INPUTS=("hotpot_train_v1.1.json;sent_drop_train")
    for input in ${INPUTS[*]}; do
        INPUT_FILE_NAME=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_PATH=$DATA_ROOT/dataset/data_raw
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_raw/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED

        echo "1. Sentence drop based data augmentation"
        # Input: INPUT_FILE
        # Output: sent_drop_json
        python dataugmentation/sentence_drop_offline.py --full_data_path $INPUT_PATH --output_data_path OUTPUT_PROCESSED --full_data_name $INPUT_FILE_NAME --drop_out 0.5

        done

}

for proc in  "preprocess"
do
    $proc
done
