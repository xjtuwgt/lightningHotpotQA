import json
import argparse
from os.path import join
import numpy as np
import math
from tqdm import tqdm
import random

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)

def hotpot_qa_sentnece_drop_examples(full_file, drop_out: float, rand_seed: int):
    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    drop_out_cases = []
    case_num = 0
    no_drop_num = 0
    for case in tqdm(full_data):

        case_id = case['_id']
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
        sent_drop_flags, drop_context, drop_supp_fact_dict, keep_sent_ids_list = sentence_drop_context(context=context, supp_fact_dict=sup_fact_dict, drop_out=drop_out)
        # print('Sum of drop flags = {}/{}'.format(sum(sent_drop_flags), len(context)))
        if sum(sent_drop_flags) == 0:
            no_drop_num = no_drop_num + 1
        case_num = case_num + 1
        case_id = case_id + "_drop_{:.2f}_seed_{}".format(drop_out, rand_seed) ## for data augmentation
        #####+++++++++++++
        case['_id'] = case_id
        drop_supp_facts = []
        for key, value in drop_supp_fact_dict.items():
            for sent_id in value:
                drop_supp_facts.append([key, sent_id])
        case['supporting_facts'] = drop_supp_facts
        case['context'] = drop_context
        if sum(sent_drop_flags) > 0:
            drop_out_cases.append(case)
        #####+++++++++++++
    print('Number of cases without drop = {}'.format(no_drop_num))
    return drop_out_cases

def sentence_drop_context(context, supp_fact_dict: dict, drop_out: float):
    sent_drop_flags = [0] * len(context)
    drop_context = []
    drop_supp_fact_dict = {}
    keep_sent_ids_list = []
    for ctx_idx, ctx in enumerate(context):
        title_i, sentences_i = ctx
        if title_i in supp_fact_dict:
            drop_ctx, drop_facts, keep_sent_ids = support_sentence_drop_out(title=title_i, sentence_list=sentences_i, drop_out=drop_out, support_fact_ids=supp_fact_dict[title_i])
            if drop_ctx is not None:
                sent_drop_flags[ctx_idx] = 1
                drop_context.append(drop_ctx)
                drop_supp_fact_dict[title_i] = drop_facts
            else:
                drop_context.append(ctx)
                drop_supp_fact_dict[title_i] = supp_fact_dict
            keep_sent_ids_list.append(keep_sent_ids)
        else:
            drop_ctx, keep_sent_ids = no_support_sentence_drop_out(title=title_i, sentence_list=sentences_i, drop_out=drop_out)
            if drop_ctx is not None:
                sent_drop_flags[ctx_idx] = 1
                drop_context.append(drop_ctx)
            else:
                drop_context.append(ctx)
            keep_sent_ids_list.append(keep_sent_ids)
    assert len(sent_drop_flags) == len(keep_sent_ids_list)
    return sent_drop_flags, drop_context, drop_supp_fact_dict, keep_sent_ids_list

def support_sentence_drop_out(title, sentence_list, support_fact_ids, drop_out):
    filtered_support_fact_ids = [_ for _ in support_fact_ids if _ < len(sentence_list)]
    sent_id_list = [s_id for s_id in range(len(sentence_list)) if s_id not in filtered_support_fact_ids]
    assert len(filtered_support_fact_ids) > 0
    assert len(sent_id_list) == (len(sentence_list) - len(filtered_support_fact_ids))
    sample_size = math.floor(len(sent_id_list) * drop_out)
    if sample_size < 1:
        keep_sent_ids = [_ for _ in range(len(sentence_list))]
        return None, None, keep_sent_ids
    keep_sent_ids = np.random.choice(sent_id_list, len(sent_id_list) - sample_size, replace=False).tolist()
    keep_sent_ids = sorted(keep_sent_ids + filtered_support_fact_ids)
    keep_sent_list = []
    new_supp_fact_ids = []
    for new_sent_idx, sent_idx in enumerate(keep_sent_ids):
        keep_sent_list.append(sentence_list[sent_idx])
        if sent_idx in filtered_support_fact_ids:
            new_supp_fact_ids.append(new_sent_idx)
    res_context = [title, keep_sent_list]
    res_support_fact_ids = new_supp_fact_ids
    assert len(res_support_fact_ids) > 0
    return res_context, res_support_fact_ids, keep_sent_ids

def no_support_sentence_drop_out(title, sentence_list, drop_out):
    sent_num = len(sentence_list)
    sample_size = math.floor(sent_num * drop_out)
    if sample_size == sent_num:
        if sent_num > 2:
            sample_size = sent_num - 2
        else:
            sample_size = 1
    if sample_size < 1:
        keep_sent_ids = [_ for _ in range(sent_num)]
        return None, keep_sent_ids
    keep_sent_ids = sorted(np.random.choice(sent_num, sent_num - sample_size, replace=False).tolist())
    keep_sent_list = [sentence_list[_] for _ in keep_sent_ids]
    res = [title, keep_sent_list]
    return res, keep_sent_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--full_data_path", type=str, required=True)
    parser.add_argument("--full_data_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help='define output data set')
    parser.add_argument("--drop_out", type=float, default=0.5, help='define dropout ratio')
    parser.add_argument("--rand_seed", type=int, default=42, help='define dropout ratio')

    args = parser.parse_args()
    for key, value in vars(args).items():
        print('Parameter {}: {}'.format(key, value))

    set_seed(random_seed=args.rand_seed)

    raw_data_file = join(args.full_data_path, args.full_data_name)
    out_put_path = args.output_path
    drop_out_ratio = args.drop_out
    drop_case_list = hotpot_qa_sentnece_drop_examples(full_file=raw_data_file, drop_out=drop_out_ratio, rand_seed=args.rand_seed)
    cached_drop_case_json_file = join(args.output_path, 'drop_sent.rand_{}.drop_{:.2f}.'.format(args.rand_seed, args.drop_out) + args.full_data_name)


    with open(cached_drop_case_json_file, 'w') as fout:
        json.dump(drop_case_list, fout)
    print('Saving {} cases into {}'.format(len(drop_case_list), cached_drop_case_json_file))

    #####
    with open(cached_drop_case_json_file, 'r', encoding='utf-8') as reader:
        drop_data = json.load(reader)
    for case in tqdm(drop_data):
        print(case)
        break