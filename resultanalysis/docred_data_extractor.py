from envs import DATASET_FOLDER
from os.path import join
import json
from tqdm import tqdm


def docred_checker():
    DOCRED_OUTPUT_PROCESSED_fill_file = join(DATASET_FOLDER, 'dataset/data_processed/docred_multihop_para.json')
    with open(DOCRED_OUTPUT_PROCESSED_fill_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    examples = []
    for case in tqdm(full_data):
        print(case)