import argparse
import json
from tqdm import tqdm

def para_ranking_preprocess(full_file, rank_file):
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    with open(rank_file, 'r', encoding='utf-8') as reader:
        rank_data = json.load(reader)

    def ranker_splitting(para_scores):
        ### split the para_scores i
        for para, score in para_scores:
            break
        return

    for case in tqdm(full_data):
        key = case['_id']
        para_rank_case = rank_data[key]


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_data', type=str, required=True)
    parser.add_argument('--rank_data', type=str, required=True)
    parser.add_argument('--split_rank_data', type=str, required=True)

    args = parser.parse_args()
    for key, value in vars(args):
        print('{}: {}'.format(key, value))
    print('*' * 100)
    full_file_name = args.full_data
    para_file_name = args.rank_data
    split_rank_file_name = args.split_rank_data