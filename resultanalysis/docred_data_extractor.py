from envs import DATASET_FOLDER
from os.path import join
import json
import collections
from tqdm import tqdm

def add_space(context_list):
    space_context = []
    for idx, context in enumerate(context_list):
        space_sent_list = []
        sent_list = context[1]
        if idx == 0:
            for sent_idx, sent in enumerate(sent_list):
                sent = sent.replace(' .', '.')
                sent = sent.replace(' ,', ',')
                sent = sent.strip()
                if sent_idx == 0:
                    space_sent_list.append(sent.strip())
                else:
                    space_sent_list.append(' ' + sent)
        else:
            for sent_idx, sent in enumerate(sent_list):
                sent = sent.replace(' .', '.')
                sent = sent.replace(' ,', ',')
                sent = sent.strip()
                space_sent_list.append(' ' + sent)
        space_context.append([context[0], space_sent_list])
    return space_context

def find_answer(answer, sents):
    for s_idx, sent in enumerate(sents):
        if answer in sent:
            return s_idx
    return -1

def find_in_answer_context(answer, context):
    founds = []
    for ctx_idx, ctx in enumerate(context):
        ans_idx = find_answer(answer=answer, sents=ctx[1])
        if ans_idx >= 0:
            founds.append(1)
            # if ctx_idx == 0:
            #     print('{} : {}: {}'.format(ctx_idx, ans_idx, len(ctx[1])))
        else:
            founds.append(0)
    ans_found_idx = -1
    assert sum(founds) <= 2
    if sum(founds) > 0:
        if founds[0] == 1:
            ans_found_idx = 0
        else:
            ans_found_idx = 1
    return ans_found_idx

def fintuner_in_answer_context(answer, context, supporting_facts):
    ans_idx = find_answer(answer=answer, sents=context[0][1])
    support_facts = set([(x[0], x[1]) for x in supporting_facts])
    if ans_idx > 0:
        if (context[0][0], ans_idx) not in support_facts:
            print(ans_idx, len(context[0][1]))
            print(supporting_facts)
            return True
    return False

def docred_refiner():
    DOCRED_OUTPUT_PROCESSED_para_file = join(DATASET_FOLDER, 'data_processed/docred/docred_multihop_para.json')
    DOCRED_OUTPUT_PROCESSED_raw_file = join(DATASET_FOLDER,
                                            'data_raw/converted_docred_total.json')  # converted_docred_total.json
    REFINEd_DOCRED_OUTPUT_PROCESSED = join(DATASET_FOLDER, 'data_raw/refined_converted_docred_total.json')
    with open(DOCRED_OUTPUT_PROCESSED_raw_file, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)
    with open(DOCRED_OUTPUT_PROCESSED_para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)
    print('loading {} data from {}'.format(len(raw_data), DOCRED_OUTPUT_PROCESSED_raw_file))
    examples = []
    answer_position = []
    answer_not_found = []
    no_answer_found = 0
    first_one_sent = 0
    title_dict = {}
    tunable_count = 0
    for case in tqdm(raw_data):
        # print(case)
        key = case['_id']
        answer = case['answer']
        context = case['context']
        support_facts = case['supporting_facts']
        title = context[0][0][:-2].strip()
        if title not in title_dict:
            title_dict[title] = 1
        else:
            title_dict[title] = title_dict[title] + 1

        fine_tune_flag = fintuner_in_answer_context(answer=answer, supporting_facts=support_facts, context=context)
        if fine_tune_flag:
            tunable_count = tunable_count + 1
        # ans_find_idx = find_in_answer_context(answer=answer, context=context)
        #         # if ans_find_idx >= 0:
        #         #     answer_position.append(ans_find_idx)
        #         # else:
        #         #     no_answer_found = no_answer_found + 1

        # if ans_find_idx == 0 and len(context[0][1]) > 1:
        #     first_one_sent = first_one_sent + 1
        # for ctx_idx, ctx in enumerate(context):
        #     is_answer_found = find_answer(answer=answer, sents=ctx[1])
        #     if is_answer_found:
        #         answer_position.append(ctx_idx)
        #         break
        #     else:
        #         continue
        # for key_name, key_value in case.items():
        #     if key_name != 'context':
        #         print('{}: {}'.format(key_name, key_value))
        #     else:
        #         for ctx_idx, ctx in enumerate(key_value):
        #             print('{}: {}'.format(ctx_idx + 1, ctx))
        # context = case['context']
        # space_context = add_space(context_list=context)
        # case['context'] = space_context
        # examples.append(case)
        # print(context)
        # print('-' * 50)
        # print(add_space(context_list=context))
        # print('*' * 100)
    print(len(raw_data))
    print(len(answer_position))
    print(sum(answer_position))
    print('no answer found = {}'.format(no_answer_found))
    print('first one sent = {}'.format(first_one_sent))
    print('tunable count = {}'.format(tunable_count))
    print('title number = {}'.format(len(title_dict)))
    # sorted_title_dict = sorted(title_dict.items(), key=lambda kv: kv[1])
    # for key, value in sorted_title_dict:
    #     print('{}: {}'.format(key, value))


def docred_checker():
    DOCRED_OUTPUT_PROCESSED_para_file = join(DATASET_FOLDER, 'data_processed/docred/docred_multihop_para.json')
    DOCRED_OUTPUT_PROCESSED_raw_file = join(DATASET_FOLDER, 'data_raw/converted_docred_total.json') #converted_docred_total.json
    # Saved_raw_DOCRED_OUTPUT_PROCESSED = join(DATASET_FOLDER, 'data_raw/space_converted_docred_total.json')
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
            if key_name != 'context':
                print('{}: {}'.format(key_name, key_value))
            else:
                for ctx_idx, ctx in enumerate(key_value):
                    print('{}: {}'.format(ctx_idx + 1, ctx))
        # context = case['context']
        # space_context = add_space(context_list=context)
        # case['context'] = space_context
        # examples.append(case)
        # print(context)
        # print('-' * 50)
        # print(add_space(context_list=context))
        print('*' * 100)
        # print('key {}'.format(key))
        # print(para_data[key])

    # json.dump(examples, open(Saved_raw_DOCRED_OUTPUT_PROCESSED, 'w'))
