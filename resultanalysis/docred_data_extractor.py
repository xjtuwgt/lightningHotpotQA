from envs import DATASET_FOLDER
from os.path import join
import json
from tqdm import tqdm


def docred_checker():
    DOCRED_OUTPUT_PROCESSED_fill_file = join(DATASET_FOLDER, 'data_raw/docred_multihop_para.json')
    with open(DOCRED_OUTPUT_PROCESSED_fill_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)
    print('loading data from {}'.format(DOCRED_OUTPUT_PROCESSED_fill_file))
    examples = []
    for case in tqdm(full_data):
        print(case)