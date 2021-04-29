import json
from tqdm import tqdm
def para_ranking_preprocess(full_file, rank_file):
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    with open(rank_file, 'r', encoding='utf-8') as reader:
        rank_data = json.load(reader)


    return