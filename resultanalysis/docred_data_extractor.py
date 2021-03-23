from envs import DATASET_FOLDER
from os.path import join
import json
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


def docred_checker():
    DOCRED_OUTPUT_PROCESSED_para_file = join(DATASET_FOLDER, 'data_processed/docred/docred_multihop_para.json')
    DOCRED_OUTPUT_PROCESSED_raw_file = join(DATASET_FOLDER, 'data_raw/converted_docred_total.json') #converted_docred_total.json
    Saved_raw_DOCRED_OUTPUT_PROCESSED = join(DATASET_FOLDER, 'data_raw/space_converted_docred_total.json')
    with open(DOCRED_OUTPUT_PROCESSED_raw_file, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)
    with open(DOCRED_OUTPUT_PROCESSED_para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)
    print('loading {} data from {}'.format(len(raw_data), DOCRED_OUTPUT_PROCESSED_raw_file))
    examples = []
    for case in tqdm(raw_data):
        # print(case)
        # key = case['_id']
        # for key_name, key_value in case.items():
        #     print('{}: {}'.format(key_name, key_value))
        context = case['context']
        space_context = add_space(context_list=context)
        case['context'] = space_context
        examples.append(case)
        print(context)
        print('-' * 50)
        print(add_space(context_list=context))
        # print('*' * 100)
        # print('key {}'.format(key))
        # print(para_data[key])





    # json.dump(examples, open(Saved_raw_DOCRED_OUTPUT_PROCESSED, 'w'))
