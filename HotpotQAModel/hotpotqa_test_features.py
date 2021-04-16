import gzip
import pickle
import json
import torch
import numpy as np
import argparse
import os

from os.path import join
from collections import Counter
from tqdm import tqdm

from model_envs import MODEL_CLASSES
from HotpotQAModel.hotpotqa_dump_features import get_cached_filename
from HotpotQAModel.hotpotqaUtils import json_loader
from eval.hotpot_evaluate_v1 import eval as hotpot_eval
from eval.hotpot_evaluate_v1 import normalize_answer

def consist_checker(para_file: str,
                    full_file: str,
                    example_file: str,
                    tokenizer,
                    data_source_type=None):
    sel_para_data = json_loader(json_file_name=para_file)
    full_data = json_loader(json_file_name=full_file)
    examples = pickle.load(gzip.open(example_file, 'rb'))
    example_dict = {e.qas_id: e for e in examples}
    assert len(sel_para_data) == len(full_data) and len(full_data) == len(examples)
    print('Number of examples = {}'.format(len(examples)))
    for row in tqdm(full_data):
        key = row['_id']
        if data_source_type is not None:
            exam_key = key + '_' + data_source_type
        else:
            exam_key = key
        print('{}\t{}'.format(key, exam_key))


    return
    # answer_dict = dict()
    # sp_dict = dict()
    # ids = list(examples.keys())
    #
    # max_sent_num = 0
    # max_entity_num = 0
    # q_type_counter = Counter()
    #
    # answer_no_match_cnt = 0
    # for i, qid in enumerate(ids):
    #     feature = features[qid]
    #     example = examples[qid]
    #     q_type = feature.ans_type
    #
    #     max_sent_num = max(max_sent_num, len(feature.sent_spans))
    #     max_entity_num = max(max_entity_num, len(feature.entity_spans))
    #     q_type_counter[q_type] += 1
    #
    #     def get_ans_from_pos(y1, y2):
    #         tok_to_orig_map = feature.token_to_orig_map
    #
    #         final_text = " "
    #         if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
    #             orig_tok_start = tok_to_orig_map[y1]
    #             orig_tok_end = tok_to_orig_map[y2]
    #
    #             ques_tok_len = len(example.question_tokens)
    #             if orig_tok_start < ques_tok_len and orig_tok_end < ques_tok_len:
    #                 ques_start_idx = example.question_word_to_char_idx[orig_tok_start]
    #                 ques_end_idx = example.question_word_to_char_idx[orig_tok_end] + len(example.question_tokens[orig_tok_end])
    #                 final_text = example.question_text[ques_start_idx:ques_end_idx]
    #             else:
    #                 orig_tok_start -= len(example.question_tokens)
    #                 orig_tok_end -= len(example.question_tokens)
    #                 ctx_start_idx = example.ctx_word_to_char_idx[orig_tok_start]
    #                 ctx_end_idx = example.ctx_word_to_char_idx[orig_tok_end] + len(example.doc_tokens[orig_tok_end])
    #                 final_text = example.ctx_text[example.ctx_word_to_char_idx[orig_tok_start]:example.ctx_word_to_char_idx[orig_tok_end]+len(example.doc_tokens[orig_tok_end])]
    #
    #         return final_text
    #         #return tokenizer.convert_tokens_to_string(tok_tokens)
    #
    #     answer_text = ''
    #     if q_type == 0 or q_type == 3:
    #         if len(feature.start_position) == 0 or len(feature.end_position) == 0:
    #             answer_text = ""
    #         else:
    #             #st, ed = example.start_position[0], example.end_position[0]
    #             #answer_text = example.ctx_text[example.ctx_word_to_char_idx[st]:example.ctx_word_to_char_idx[ed]+len(example.doc_tokens[example.end_position[0]])]
    #             answer_text = get_ans_from_pos(feature.start_position[0], feature.end_position[0])
    #             if normalize_answer(answer_text) != normalize_answer(example.orig_answer_text):
    #                 print("{} | {} | {} | {} | {}".format(qid, answer_text, example.orig_answer_text, feature.start_position[0], feature.end_position[0]))
    #                 answer_no_match_cnt += 1
    #         if q_type == 3 and use_ent_ans:
    #             ans_id = feature.answer_in_entity_ids[0]
    #             st, ed = feature.entity_spans[ans_id]
    #             answer_text = get_ans_from_pos(st, ed)
    #     elif q_type == 1:
    #         answer_text = 'yes'
    #     elif q_type == 2:
    #         answer_text = 'no'
    #
    #     answer_dict[qid] = answer_text
    #     cur_sp = []
    #     for sent_id in feature.sup_fact_ids:
    #         cur_sp.append(example.sent_names[sent_id])
    #     sp_dict[qid] = cur_sp
    #
    # final_pred = {'answer': answer_dict, 'sp': sp_dict}
    # json.dump(final_pred, open(pred_file, 'w'))
    #
    # print("Maximum sentence num: {}".format(max_sent_num))
    # print("Maximum entity num: {}".format(max_entity_num))
    # print("Question type: {}".format(q_type_counter))
    # print("Answer doesnot match: {}".format(answer_no_match_cnt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--para_path", type=str, required=True)
    parser.add_argument("--full_data", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help='define output directory')

    # Other parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--filter_no_ans", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--ranker", default=None, type=str, required=True,
                        help="The ranker for paragraph ranking")
    parser.add_argument("--reverse", action='store_true',
                        help="Set this flag if you are using reverse data.")

    args = parser.parse_args()
    print('*' * 75)
    for key, value in vars(args).items():
        print('Hype-parameter: {}:\t{}'.format(key, value))
    print('*' * 75)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)

    ranker = args.ranker
    data_type = args.data_type
    if args.do_lower_case:
        ranker = ranker + '_low'
    data_source_name = "{}".format(ranker)
    if "train" in data_type:
        data_source_type = data_source_name
    else:
        data_source_type = None
    print('data_type = {} \n data_source_id= {} \n data_source_name = {}'.format(data_type, data_source_type,
                                                                                 data_source_name))
    cached_examples_file = os.path.join(args.output_dir,
                                        get_cached_filename('{}_hotpotqa_tokenized_examples'.format(data_source_name), args))
    consist_checker(para_file=args.para_path, full_file=args.full_data, example_file=cached_examples_file, tokenizer=tokenizer)
