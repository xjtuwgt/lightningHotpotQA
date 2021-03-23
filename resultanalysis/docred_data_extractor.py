from envs import DATASET_FOLDER
from os.path import join
import json
from tqdm import tqdm


def docred_checker():
    DOCRED_OUTPUT_PROCESSED_para_file = join(DATASET_FOLDER, 'data_processed/docred/docred_multihop_para.json')
    DOCRED_OUTPUT_PROCESSED_raw_file = join(DATASET_FOLDER, 'data_raw/converted_docred_total.json')
    with open(DOCRED_OUTPUT_PROCESSED_raw_file, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)
    with open(DOCRED_OUTPUT_PROCESSED_para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)
    print('loading {} data from {}'.format(len(raw_data), DOCRED_OUTPUT_PROCESSED_raw_file))
    examples = []
    for case in tqdm(raw_data):
        # print(case)
        key = case['_id']
        for key_name, key_value in case.items():
            print('{}: {}'.format(key_name, key_value))
        print('*' * 100)
        # print('key {}'.format(key))
        # print(para_data[key])