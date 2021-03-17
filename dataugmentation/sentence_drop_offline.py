import json
import argparse
from os.path import join
import numpy as np
import math
from tqdm import tqdm

def hotpot_qa_sentnece_drop_examples(full_file, drop_out: float):
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    drop_out_cases = []
    for case in tqdm(full_data):
        sup_facts = list(set([(sp[0], sp[1]) for sp in case['supporting_facts']]))
        sup_fact_dict = {}
        for sp in sup_facts:
            if sp[0] not in sup_fact_dict:
                sup_fact_dict[sp[0]] = [sp[1]]
            else:
                sup_fact_dict[sp[0]].append(sp[1])
        for key, value in sup_fact_dict.items():
            sup_fact_dict[key] = sorted(value)

        context = case['context']
        assert len(context) >= 2
        ##############################################

def sentence_drop_context(context, supp_fact_dict: dict, drop_out: float):
    sent_drop_flags = [0] * len(context)
    drop_context = []
    for ctx_idx, ctx in enumerate(context):
        title_i, sentences_i = ctx
        if title_i not in supp_fact_dict:
            no_support_sentence_drop_out(title=title_i, sentence_list=sentences_i, drop_out=drop_out)
        else:
            drop_ctx = no_support_sentence_drop_out(title=title_i, sentence_list=sentences_i, drop_out=drop_out)
            if drop_ctx is not None:
                sent_drop_flags[ctx_idx] = 1
            else:
                print(len(sentences_i))
    return sent_drop_flags

def support_sentence_drop_out(title, sentence_list, drop_out, support_fact_ids):

    return


def no_support_sentence_drop_out(title, sentence_list, drop_out):
    sent_num = len(sentence_list)
    sample_size = math.floor(sent_num * drop_out)
    print(sent_num)
    if sample_size < 1:
        return None
    drop_sent_ids = np.random.choice(sent_num, sample_size, replace=False).tolist()
    sent_keep_list = [sent_ for sent_idx, sent_ in enumerate(sentence_list) if sent_idx not in drop_sent_ids]
    res = [title, sent_keep_list]
    return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--full_data_path", type=str, required=True)
    parser.add_argument("--full_data_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help='define output data set')
    parser.add_argument("--drop_out", type=float, default=0.5, help='define dropout ratio')

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('Parameter {}: {}'.format(key, value))

    raw_data_file = join(args.full_data_path, args.full_data_name)
    out_put_path = args.output_path
    drop_out_ratio = args.drop_out
    hotpot_qa_sentnece_drop_examples(full_file=raw_data_file, drop_out=drop_out_ratio)