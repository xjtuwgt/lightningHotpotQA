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
import itertools

from model_envs import MODEL_CLASSES
from jdqamodel.hotpotqa_dump_features import get_cached_filename
from jdqamodel.hotpotqaUtils import json_loader
from jdqamodel.hotpotqa_data_structure import Example
from jdqamodel.hotpotqa_data_loader import case_to_features, example_sent_drop, trim_input_span
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
    no_answer_count = 0
    for row in tqdm(full_data):
        key = row['_id']
        if data_source_type is not None:
            exam_key = key + '_' + data_source_type
        else:
            exam_key = key
        raw_question = row['question']
        raw_context = row['context']
        raw_answer = row['answer']
        example_i: Example = example_dict[exam_key]
        exm_question = example_i.question_text
        exm_answer = example_i.answer_text
        exm_context = example_i.ctx_text
        exm_ctx_token_list = example_i.ctx_tokens
        exm_ctx_input_ids = example_i.ctx_input_ids
        # print('{}\t{}'.format(key, exam_key))
        # print('raw question:', raw_question)
        # print('exm question:', exm_question)
        # print('raw answer:', raw_answer)
        # print('exm answer:', exm_answer)
        answer_positions = example_i.answer_positions
        para_names = example_i.para_names
        para_name_dict = dict([(x[1], x[0]) for x in enumerate(para_names)])
        encode_answer = ''
        for para_i, sent_i, start_i, end_i in answer_positions:
            para_idx = para_name_dict[para_i]
            sent_ids = exm_ctx_input_ids[para_idx][sent_i]
            encode_answer = tokenizer.decode(sent_ids[start_i:end_i])
        # if raw_answer in ['yes', 'no']:
        #     print('{}\t{}\t{}\t{}'.format(raw_answer, exm_answer, encode_answer, example_i.ctx_with_answer))
        if exm_answer in ['noanswer']:
            print('{}\t{}\tencode:{}\t{}'.format(raw_answer, exm_answer, encode_answer, example_i.ctx_with_answer))
        if not example_i.ctx_with_answer and raw_answer not in ['yes', 'no']:
            no_answer_count = no_answer_count + 1

        contex_text = []
        ctx_dict = dict(raw_context)
        contex_text = []
        for para_name in para_names:
            contex_text.append(ctx_dict[para_name])

        for para_idx, ctx_token_list in enumerate(exm_ctx_token_list):
            ctx_inp_id_list = exm_ctx_input_ids[para_idx]
            orig_context = contex_text[para_idx]
            for sent_idx, sent_inp_ids in enumerate(ctx_inp_id_list):
                print(ctx_token_list[sent_idx])
                print(tokenizer.decode(sent_inp_ids))
                print(orig_context[sent_idx])
        print('*' * 75)

        # if exm_answer.strip() in ['noanswer']:
        #     print('raw answer:', raw_answer)
        #     print('exm answer:', exm_answer)
        #     no_answer_count = no_answer_count + 1
        #     print('raw context:', raw_context)
        #     print('*' * 75)
        #     print('exm context:', exm_context)
        #     print('*' * 75)
        #     print('exm tokens: ', exm_ctx_token_list)
        #     print('*' * 75)
        #     for x in exm_ctx_input_ids:
        #         for y in x:
        #             print('exm decode: ', tokenizer.decode(y))

    print(no_answer_count)
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

def case_to_feature_checker(para_file: str,
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
    no_answer_count = 0
    sep_id = tokenizer.encode(tokenizer.sep_token)
    print(sep_id)

    ans_count_list = []
    for row in tqdm(full_data):
        key = row['_id']
        if data_source_type is not None:
            exam_key = key + '_' + data_source_type
        else:
            exam_key = key
        example_i: Example = example_dict[exam_key]
    #     sel_para_names = sel_para_data[key]
    #     print('selected para names: ', sel_para_names)
    #     print('example para names: ', example_i.para_names)
    #     doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = \
    #         case_to_features(case=example_i, train_dev=True)
    #     orig_query = row['question']
    #     query_input_ids = doc_input_ids[query_spans[0][0]:query_spans[0][1]]
    #     decoded_query = tokenizer.decode(query_input_ids)
    #     # print('Orig query = {}'.format(orig_query))
    #     # print('Decoded query = {}'.format(decoded_query))
    #     # print('para number {}'.format(len(para_spans)))
    #     # print('sent number {}'.format(len(sent_spans)))
    #     # print('ans_spans number {}'.format(len(ans_spans)))
    #     orig_answer = row['answer']
    #     exm_answer = example_i.answer_text
    #     ##+++++++
    #     all_sents = []
    #     ctx_dict = dict(row['context'])
    #     contex_text = []
    #     for para_name in example_i.para_names:
    #         contex_text.append(ctx_dict[para_name])
    #         all_sents += ctx_dict[para_name]
    #
    #
    #     # for s_idx, sent_span in enumerate(sent_spans):
    #     #     sent_inp_ids = doc_input_ids[sent_span[0]:sent_span[1]]
    #     #     # print(sent_inp_ids)
    #     #     decoded_sent = tokenizer.decode(sent_inp_ids)
    #     #     print('{} orig sent: {}'.format(s_idx, all_sents[s_idx]))
    #     #     print('{} deco sent: {}'.format(s_idx, decoded_sent))
    #     #     print('$' * 10)
    #     # print('-' * 75)
    #
    #
    #     for ans_idx, ans_span in enumerate(ans_spans):
    #         # print(ans_span)
    #         # print(len(doc_input_ids))
    #         # if ans_span[0] < 0 or ans_span[0] >= len(doc_input_ids) or ans_span[1] >= len(doc_input_ids):
    #         #     print(ans_span)
    #         #     print(len(doc_input_ids))
    #         ans_inp_ids = doc_input_ids[ans_span[0]:ans_span[1]]
    #         decoded_ans = tokenizer.decode(ans_inp_ids)
    #         print('{} Orig\t{}\t{}\t{}'.format(ans_idx, orig_answer, exm_answer, decoded_ans))
    #     print('*' * 75)
    #
    #     # for p_idx, para_span in enumerate(para_spans):
    #     #     para_inp_ids = doc_input_ids[para_span[0]:para_span[1]]
    #     #     decoded_para = tokenizer.decode(para_inp_ids)
    #     #     print('{} orig para: {}'.format(p_idx, contex_text[p_idx]))
    #     #     print('{} deco para: {}'.format(p_idx, decoded_para))
    #     print('-' * 75)
    #     ans_count_list.append(len(ans_spans))
    #
    # print('Sum of ans count = {}'.format(sum(ans_count_list)))
def sent_drop_case_to_feature_checker(para_file: str,
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
    no_answer_count = 0
    sep_id = tokenizer.encode(tokenizer.sep_token)
    print(sep_id)

    ans_count_list = []
    one_supp_sent = 0
    miss_supp_count = 0
    larger_512 = 0
    drop_larger_512 = 0
    max_query_len = 0
    query_len_list = []
    for row in tqdm(full_data):
        key = row['_id']
        if data_source_type is not None:
            exam_key = key + '_' + data_source_type
        else:
            exam_key = key
        example_i: Example = example_dict[exam_key]
        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = \
            case_to_features(case=example_i, train_dev=True)
        # # print(len(doc_input_ids))
        # # print('orig', doc_input_ids)
        # # print(len(sent_spans))
        if len(doc_input_ids) > 512:
            larger_512 += 1
        # print('orig', example_i.ctx_input_ids)
        drop_example_i = example_sent_drop(case=example_i, drop_ratio=1.0)
        # print('drop', drop_example_i.ctx_input_ids)
        query_len_list.append(query_spans[0][1])
        if max_query_len < query_spans[0][1]:
            max_query_len = query_spans[0][1]
            query_len_list.append(query_spans[0][1])
            print(max_query_len)

        # print('orig q ids {}'.format(example_i.question_input_ids))
        # print('drop q ids {}'.format(drop_example_i.question_input_ids))
        # supp_para_names = list(set([x[0] for x in row['supporting_facts']]))
        # exam_para_names = [example_i.para_names[x] for x in example_i.sup_para_id]
        # drop_exam_para_names = [drop_example_i.para_names[x] for x in drop_example_i.sup_para_id]
        # print('drop', example_i.para_names)
        # print(drop_example_i.para_names)

        # print('orig {}'.format(supp_para_names))
        # print('exam {}'.format(exam_para_names))
        # print('drop exam {}'.format(drop_exam_para_names))
        #
        #
        # print(example_i.sent_num, drop_example_i.sent_num)
        # orig_supp_count = len(row['supporting_facts'])
        # if drop_example_i.sent_num < orig_supp_count:
        #     miss_supp_count +=1
        # if drop_example_i.sent_num < 2:
        #     one_supp_sent += 1
        # sel_para_names = sel_para_data[key]
        # print('selected para names: ', sel_para_names)
        # print('example para names: ', example_i.para_names)

        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = \
            case_to_features(case=drop_example_i, train_dev=True)
        # print(len(drop_doc_input_ids))
        # # print('drop', drop_doc_input_ids)
        # print(len(drop_sent_spans))
        if len(doc_input_ids) > 512:
            drop_larger_512 += 1

        # print(type(doc_input_ids), type(query_spans), type(para_spans), type(sent_spans), type(ans_spans))
        # orig_query = row['question']
        # query_input_ids = doc_input_ids[query_spans[0][0]:query_spans[0][1]]
        # decoded_query = tokenizer.decode(query_input_ids)
        # # print('Orig query = {}'.format(orig_query))
        # # print('Decoded query = {}'.format(decoded_query))
        # # print('para number {}'.format(len(para_spans)))
        # # print('sent number {}'.format(len(sent_spans)))
        # # print('ans_spans number {}'.format(len(ans_spans)))
        orig_answer = row['answer']
        exm_answer = example_i.answer_text
        # print('{}\t{}'.format(exm_answer, ans_type_label))
        #
        # assert len(example_i.sup_para_id) == len(drop_example_i.sup_para_id)
        # assert len(example_i.sup_fact_id) == len(drop_example_i.sup_fact_id)
        # ##+++++++
        all_sents = []
        ctx_dict = dict(row['context'])
        contex_text = []
        for para_name in example_i.para_names:
            contex_text.append(ctx_dict[para_name])
            all_sents += ctx_dict[para_name]


        # for s_idx, sent_span in enumerate(sent_spans):
        #     sent_inp_ids = doc_input_ids[sent_span[0]:sent_span[1]]
        #     # print(sent_inp_ids)
        #     decoded_sent = tokenizer.decode(sent_inp_ids)
        #     print('{} orig sent: {}'.format(s_idx, all_sents[s_idx]))
        #     print('{} deco sent: {}'.format(s_idx, decoded_sent))
        #     print('$' * 10)
        # print('-' * 75)


        # for ans_idx, ans_span in enumerate(ans_spans):
        #     # print(ans_span)
        #     # print(len(doc_input_ids))
        #     # if ans_span[0] < 0 or ans_span[0] >= len(doc_input_ids) or ans_span[1] >= len(doc_input_ids):
        #     #     print(ans_span)
        #     #     print(len(doc_input_ids))
        #     ans_inp_ids = doc_input_ids[ans_span[0]:ans_span[1]]
        #     decoded_ans = tokenizer.decode(ans_inp_ids)
        #     print('{} Orig\t{}\t{}\t{}\t{}'.format(ans_idx, orig_answer, exm_answer, decoded_ans, ans_type_label[0]))
        # print('*' * 75)
        #
        # # for p_idx, para_span in enumerate(para_spans):
        # #     para_inp_ids = doc_input_ids[para_span[0]:para_span[1]]
        # #     decoded_para = tokenizer.decode(para_inp_ids)
        # #     print('{} orig para: {}'.format(p_idx, contex_text[p_idx]))
        # #     print('{} deco para: {}'.format(p_idx, decoded_para))
        # # print('-' * 75)
        #
        ans_count_list.append(len(ans_spans))

    print('Sum of ans count = {}'.format(sum(ans_count_list)))
    print('One support sent count = {}'.format(one_supp_sent))
    print('Miss support sent count = {}'.format(miss_supp_count))
    print('Larger than 512 count = {}'.format(larger_512))
    print('Larger than 512 count after drop = {}'.format(drop_larger_512))
    print('Max query len = {}'.format(max_query_len))
    query_len_array = np.array(query_len_list)

    print('99 = {}'.format(np.percentile(query_len_array, 99)))
    print('97.5 = {}'.format(np.percentile(query_len_array, 97.5)))


def trim_case_to_feature_checker(para_file: str,
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
    no_answer_count = 0
    trim_no_answer_count = 0
    sep_id = tokenizer.encode(tokenizer.sep_token)
    print(sep_id)

    ans_count_list = []
    one_supp_sent = 0
    miss_supp_count = 0
    larger_512 = 0
    drop_larger_512 = 0
    max_query_len = 0
    query_len_list = []


    for row in tqdm(full_data):
        key = row['_id']
        if data_source_type is not None:
            exam_key = key + '_' + data_source_type
        else:
            exam_key = key
        example_i: Example = example_dict[exam_key]
        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = \
            case_to_features(case=example_i, train_dev=True)
        # # print(len(doc_input_ids))
        # # print('orig', doc_input_ids)
        # # print(len(sent_spans))
        if len(doc_input_ids) > 512:
            larger_512 += 1
        # print('orig', example_i.ctx_input_ids)
        drop_example_i = example_sent_drop(case=example_i, drop_ratio=0.25)
        # print('drop', drop_example_i.ctx_input_ids)
        query_len_list.append(query_spans[0][1])
        if max_query_len < query_spans[0][1]:
            max_query_len = query_spans[0][1]
            query_len_list.append(query_spans[0][1])
            print(max_query_len)

        # print('orig q ids {}'.format(example_i.question_input_ids))
        # print('drop q ids {}'.format(drop_example_i.question_input_ids))
        # supp_para_names = list(set([x[0] for x in row['supporting_facts']]))
        # exam_para_names = [example_i.para_names[x] for x in example_i.sup_para_id]
        # drop_exam_para_names = [drop_example_i.para_names[x] for x in drop_example_i.sup_para_id]
        # print('drop', example_i.para_names)
        # print(drop_example_i.para_names)

        # print('orig {}'.format(supp_para_names))
        # print('exam {}'.format(exam_para_names))
        # print('drop exam {}'.format(drop_exam_para_names))
        #
        #
        # print(example_i.sent_num, drop_example_i.sent_num)
        # orig_supp_count = len(row['supporting_facts'])
        # if drop_example_i.sent_num < orig_supp_count:
        #     miss_supp_count +=1
        # if drop_example_i.sent_num < 2:
        #     one_supp_sent += 1
        # sel_para_names = sel_para_data[key]
        # print('selected para names: ', sel_para_names)
        # print('example para names: ', example_i.para_names)

        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = \
            case_to_features(case=drop_example_i, train_dev=True)
        # print(len(drop_doc_input_ids))
        # # print('drop', drop_doc_input_ids)
        # print(len(drop_sent_spans))
        if len(doc_input_ids) > 512:
            drop_larger_512 += 1


        # print(type(doc_input_ids), type(query_spans), type(para_spans), type(sent_spans), type(ans_spans))
        # orig_query = row['question']
        # query_input_ids = doc_input_ids[query_spans[0][0]:query_spans[0][1]]
        # decoded_query = tokenizer.decode(query_input_ids)
        # # print('Orig query = {}'.format(orig_query))
        # # print('Decoded query = {}'.format(decoded_query))
        # # print('para number {}'.format(len(para_spans)))
        # # print('sent number {}'.format(len(sent_spans)))
        # # print('ans_spans number {}'.format(len(ans_spans)))
        # orig_answer = row['answer']
        # exm_answer = example_i.answer_text
        # print('{}\t{}'.format(exm_answer, ans_type_label))
        #
        # assert len(example_i.sup_para_id) == len(drop_example_i.sup_para_id)
        # assert len(example_i.sup_fact_id) == len(drop_example_i.sup_fact_id)
        # ##+++++++
        all_sents = []
        ctx_dict = dict(row['context'])
        contex_text = []
        for para_name in example_i.para_names:
            contex_text.append(ctx_dict[para_name])
            all_sents += ctx_dict[para_name]

        if ans_type_label == 2 and len(ans_spans) == 0:
            no_answer_count = no_answer_count + 1

        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans = trim_input_span(doc_input_ids, query_spans, para_spans, sent_spans,
                                                                                        limit=512, sep_token_id=tokenizer.sep_token_id, ans_spans=ans_spans)

        if ans_type_label == 2 and len(ans_spans) == 0:
            trim_no_answer_count = trim_no_answer_count + 1

        # for s_idx, sent_span in enumerate(sent_spans):
        #     sent_inp_ids = doc_input_ids[sent_span[0]:sent_span[1]]
        #     # print(sent_inp_ids)
        #     decoded_sent = tokenizer.decode(sent_inp_ids)
        #     print('{} orig sent: {}'.format(s_idx, all_sents[s_idx]))
        #     print('{} deco sent: {}'.format(s_idx, decoded_sent))
        #     print('$' * 10)
        # print('-' * 75)


        # for ans_idx, ans_span in enumerate(ans_spans):
        #     # print(ans_span)
        #     # print(len(doc_input_ids))
        #     # if ans_span[0] < 0 or ans_span[0] >= len(doc_input_ids) or ans_span[1] >= len(doc_input_ids):
        #     #     print(ans_span)
        #     #     print(len(doc_input_ids))
        #     ans_inp_ids = doc_input_ids[ans_span[0]:ans_span[1]]
        #     decoded_ans = tokenizer.decode(ans_inp_ids)
        #     print('{} Orig\t{}\t{}\t{}\t{}'.format(ans_idx, orig_answer, exm_answer, decoded_ans, ans_type_label[0]))
        # print('*' * 75)
        #
        # # for p_idx, para_span in enumerate(para_spans):
        # #     para_inp_ids = doc_input_ids[para_span[0]:para_span[1]]
        # #     decoded_para = tokenizer.decode(para_inp_ids)
        # #     print('{} orig para: {}'.format(p_idx, contex_text[p_idx]))
        # #     print('{} deco para: {}'.format(p_idx, decoded_para))
        # # print('-' * 75)
        #
        ans_count_list.append(len(ans_spans))

    # print('Sum of ans count = {}'.format(sum(ans_count_list)))
    # print('One support sent count = {}'.format(one_supp_sent))
    # print('Miss support sent count = {}'.format(miss_supp_count))
    # print('Larger than 512 count = {}'.format(larger_512))
    # print('Larger than 512 count after drop = {}'.format(drop_larger_512))
    # print('Max query len = {}'.format(max_query_len))
    # query_len_array = np.array(query_len_list)
    #
    # print('99 = {}'.format(np.percentile(query_len_array, 99)))
    # print('97.5 = {}'.format(np.percentile(query_len_array, 97.5)))

    print('No answer count = {}'.format(no_answer_count))
    print('Trim no answer count = {}'.format(trim_no_answer_count))

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
    # consist_checker(para_file=args.para_path, full_file=args.full_data, example_file=cached_examples_file, tokenizer=tokenizer, data_source_type=data_source_type)

    # case_to_feature_checker(para_file=args.para_path, full_file=args.full_data, example_file=cached_examples_file,
    #                 tokenizer=tokenizer, data_source_type=data_source_type)
    # sent_drop_case_to_feature_checker(para_file=args.para_path, full_file=args.full_data, example_file=cached_examples_file,
    #                         tokenizer=tokenizer, data_source_type=data_source_type)

    trim_case_to_feature_checker(para_file=args.para_path, full_file=args.full_data,
                                      example_file=cached_examples_file,
                                      tokenizer=tokenizer, data_source_type=data_source_type)
