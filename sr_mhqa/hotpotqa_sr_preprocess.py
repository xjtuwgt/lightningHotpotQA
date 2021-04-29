import argparse
from sr_mhqa.hotpotqa_sr_data_structure import Example
from sr_mhqa.hotpotqa_sr_utils import hotpot_answer_neg_sents_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_data', type=str, required=True)
    parser.add_argument('--split_rank_data', type=str, required=True)