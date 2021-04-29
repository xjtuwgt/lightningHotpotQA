import argparse
import json
from time import time
from tqdm import tqdm

def para_ranking_preprocess(full_file, rank_file):
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)
    print('Loading {} records from {}'.format(len(full_data), full_file))

    with open(rank_file, 'r', encoding='utf-8') as reader:
        rank_data = json.load(reader)
    print('Loading {} records from {}'.format(len(rank_data), rank_file))

    def ranker_splitting(para_scores):
        ### split the para_scores i
        top4 = para_scores[:4]
        if len(para_scores) <= 4:
            return (top4, [], [])
        if len(para_scores) < 8:
            top4_8 = para_scores[4:]
            return (top4, top4_8, [])
        else:
            top4_8 = para_scores[4:8]
            top8_plus = para_scores[8:]
            return (top4, top4_8, top8_plus)

        # for para, score in para_scores:
        #     break
        # return
    split_rank_dict = {}
    for case in tqdm(full_data):
        key = case['_id']
        para_rank_case = rank_data[key]
        para_split_row = ranker_splitting(para_scores=para_rank_case)
        print(para_rank_case)
        print(para_split_row)
        break


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_data', type=str, required=True)
    parser.add_argument('--rank_data', type=str, required=True)
    parser.add_argument('--split_rank_data', type=str, required=True)

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))
    print('*' * 100)
    full_file_name = args.full_data
    para_rank_file_name = args.rank_data
    split_rank_file_name = args.split_rank_data
    start_time = time()
    para_ranking_preprocess(full_file=full_file_name, rank_file=para_rank_file_name)
    print('Data splitting takes {:.4f} seconds'.format(time() - start_time))