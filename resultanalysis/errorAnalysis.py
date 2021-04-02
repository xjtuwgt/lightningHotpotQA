import gzip
import pickle
import json
import torch
import numpy as np
import argparse
import os
from sklearn.metrics import confusion_matrix
from eval.hotpot_evaluate_v1 import normalize_answer, exact_match_score, f1_score

from os.path import join
from collections import Counter
from longformerscripts.longformerIREvaluation import recall_computation

from model_envs import MODEL_CLASSES
from plmodels.pldata_processing import Example, InputFeatures, get_cached_filename

def exmple_infor_collection(example: Example):
    # Example
    # self.qas_id = qas_id
    #         self.qas_type = qas_type
    #         self.question_tokens = question_tokens
    #         self.doc_tokens = doc_tokens
    #         self.question_text = question_text
    #         self.sent_num = sent_num
    #         self.sent_names = sent_names
    #         self.para_names = para_names
    #         self.sup_fact_id = sup_fact_id
    #         self.sup_para_id = sup_para_id
    #         self.ques_entities_text = ques_entities_text
    #         self.ctx_entities_text = ctx_entities_text
    #         self.para_start_end_position = para_start_end_position
    #         self.sent_start_end_position = sent_start_end_position
    #         self.ques_entity_start_end_position = ques_entity_start_end_position
    #         self.ctx_entity_start_end_position = ctx_entity_start_end_position
    #         self.question_word_to_char_idx = question_word_to_char_idx
    #         self.ctx_text = ctx_text
    #         self.ctx_word_to_char_idx = ctx_word_to_char_idx
    #         self.edges = edges
    #         self.orig_answer_text = orig_answer_text
    #         self.answer_in_ques_entity_ids = answer_in_ques_entity_ids
    #         self.answer_in_ctx_entity_ids = answer_in_ctx_entity_ids
    #         self.answer_candidates_in_ctx_entity_ids= answer_candidates_in_ctx_entity_ids
    #         self.start_position = start_position
    #         self.end_position = end_position
    doc_tokens = example.doc_tokens
    query_tokens = example.question_tokens
    sent_num = example.sent_num
    sent_start_end_position = example.sent_start_end_position
    ent_start_end_position = example.ctx_entity_start_end_position
    print(sent_num, len(sent_start_end_position))

    return

def feature_infor_collection(feature: InputFeatures):
    instance_variables = vars(feature)
    for key, value in instance_variables.items():
        print(key)

    # print(instance_variables)
    # features
    # self.qas_id = qas_id
    #         self.doc_tokens = doc_tokens
    #         self.doc_input_ids = doc_input_ids
    #         self.doc_input_mask = doc_input_mask
    #         self.doc_segment_ids = doc_segment_ids
    #
    #         self.query_tokens = query_tokens
    #         self.query_input_ids = query_input_ids
    #         self.query_input_mask = query_input_mask
    #         self.query_segment_ids = query_segment_ids
    #
    #         self.para_spans = para_spans
    #         self.sent_spans = sent_spans
    #         self.entity_spans = entity_spans
    #         self.q_entity_cnt = q_entity_cnt
    #         self.sup_fact_ids = sup_fact_ids
    #         self.sup_para_ids = sup_para_ids
    #         self.ans_type = ans_type
    #
    #         self.edges = edges
    #         self.token_to_orig_map = token_to_orig_map
    #         self.orig_answer_text = orig_answer_text
    #         self.answer_in_entity_ids = answer_in_entity_ids
    #         self.answer_candidates_ids = answer_candidates_ids
    #
    #         self.start_position = start_position
    #         self.end_position = end_position
    return

def set_comparison(prediction_list, true_list):
    def em():
        if len(prediction_list) != len(true_list):
            return False
        for pred in prediction_list:
            if pred not in true_list:
                return False
        return True
    if em():
        return 'em'

    is_empty_set = len(set(prediction_list).intersection(set(true_list))) == 0
    if is_empty_set:
        return 'no_over_lap'
    is_subset = set(true_list).issubset(set(prediction_list))
    if is_subset:
        return 'super_of_gold'
    is_super_set = set(prediction_list).issubset(set(true_list))
    if is_super_set:
        return 'sub_of_gold'
    return 'others'

def data_analysis(raw_data, examples, features, tokenizer, use_ent_ans=False):
    # example_sent_num_list = []
    # example_ent_num_list = []
    # example_ctx_num_list = []
    example_doc_recall_list = []
    feature_doc_recall_list = []

    example_sent_recall_list = []
    feature_sent_recall_list = []
    trim_yes_no_count = 0
    for row in raw_data:
        qid = row['_id']
        answer = row['answer']
        gold_doc_names = list(set([_[0] for _ in row['supporting_facts']]))
        raw_context = row['context']
        raw_supp_sents = [(x[0], x[1]) for x in row['supporting_facts']]
        ################################################################################################################
        feature = features[qid]
        feature_dict = vars(feature)
        doc_input_ids = feature_dict['doc_input_ids']
        assert len(doc_input_ids) == 512
        # doc_512_context = tokenizer.decode(doc_input_ids, skip_special_tokens=True)
        para_spans = feature_dict['para_spans']
        trim_doc_names = [_[2] for _ in para_spans]
        feature_em_recall = recall_computation(prediction=trim_doc_names, gold=gold_doc_names)
        feature_doc_recall_list.append(feature_em_recall)
        trim_sent_spans = feature_dict['sent_spans']

        ################################################################################################################
        # for key, value in feature_dict.items():
        #     print('F: {}\t{}'.format(key, value))
        ################################################################################################################
        example = examples[qid]
        example_dict = vars(example)
        example_doc_names = example_dict['para_names']
        em_recall = recall_computation(prediction=example_doc_names, gold=gold_doc_names)
        example_doc_recall_list.append(em_recall)
        example_sent_names = example_dict['sent_names']
        em_sent_recall = recall_computation(prediction=example_sent_names, gold=raw_supp_sents)
        example_sent_recall_list.append(em_sent_recall)
        trim_span_sent_names = [example_sent_names[i] for i in range(len(trim_sent_spans))]
        trim_em_sent_recall = recall_computation(prediction=trim_span_sent_names, gold=raw_supp_sents)
        feature_sent_recall_list.append(trim_em_sent_recall)
        if trim_em_sent_recall != 1:
            if answer in ['yes', 'no']:
                trim_yes_no_count += 1
        ################################################################################################################
        # for key, value in example_dict.items():
        #     print('E:{}\t{}'.format(key, value))
        ################################################################################################################
        # print(len(example_doc_names), len(para_spans))
        # if len(example_doc_names) > len(para_spans):
        #     print(qid)
        #     print('Example context:\n{}'.format(example_dict['ctx_text']))
        #     print('-' * 100)
        #     print('Feature context:\n{}'.format(tokenizer.decode(doc_input_ids, skip_special_tokens=True)))
        #     print('+' * 100)
        #     cut_para_names = [x for x in example_doc_names if x not in trim_doc_names]
        #     print(len(example_doc_names), len(para_spans), len(cut_para_names))
        #     for c_idx, cut_para in enumerate(cut_para_names):
        #         for ctx_idx, ctx in enumerate(raw_context):
        #             if cut_para == ctx[0]:
        #                 print('Cut para {}:\n{}'.format(c_idx, ctx[1]))
        #     print('*'*100)

        # print('$' * 100)
        # if len(example_sent_names) > len(trim_sent_spans):
        #     print(qid)
        #     break

    print('Example doc recall: {}'.format(sum(example_doc_recall_list)/len(example_doc_recall_list)))
    print('Example doc recall (512 trim): {}'.format(sum(feature_doc_recall_list)/len(feature_doc_recall_list)))
    print('Example sent recall: {}'.format(sum(example_sent_recall_list) / len(example_sent_recall_list)))
    print('Example sent recall (512 trim): {}'.format(sum(feature_sent_recall_list) / len(feature_sent_recall_list)))
    print('Trim yes no : {}'.format(trim_yes_no_count))

def error_analysis(raw_data, predictions, tokenizer, use_ent_ans=False):
    yes_no_span_predictions = []
    yes_no_span_true = []
    prediction_ans_type_counter = Counter()
    prediction_sent_type_counter = Counter()
    prediction_para_type_counter = Counter()

    pred_ans_type_list = []
    pred_sent_type_list = []
    pred_doc_type_list = []
    pred_sent_count_list = []

    pred_para_count_list = []
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for row in raw_data:
        qid = row['_id']
        sp_predictions = predictions['sp'][qid]
        sp_predictions = [(x[0], x[1]) for x in sp_predictions]
        ans_prediction = predictions['answer'][qid]

        raw_answer = row['answer']
        raw_answer = normalize_answer(raw_answer)
        ans_prediction = normalize_answer(ans_prediction)
        sp_golds = row['supporting_facts']
        sp_golds = [(x[0], x[1]) for x in sp_golds]
        sp_para_golds = list(set([_[0] for _ in sp_golds]))
        ##+++++++++++
        # sp_predictions = [x for x in sp_predictions if x[0] in sp_para_golds]
        # sp_predictions
        print("{}\t{}\t{}".format(qid, len(set(sp_golds)), len(set(sp_predictions))))
        sp_para_predictions = list(set([x[0] for x in sp_predictions]))
        pred_para_count_list.append(len(sp_para_predictions))
        # +++++++++++
        if len(set(sp_golds)) > len(set(sp_predictions)):
            pred_sent_count_list.append('less')
        elif len(set(sp_golds)) < len(set(sp_predictions)):
            pred_sent_count_list.append('more')
        else:
            pred_sent_count_list.append('equal')
        ##+++++++++++
        sp_sent_type = set_comparison(prediction_list=sp_predictions, true_list=sp_golds)
        ###+++++++++
        prediction_sent_type_counter[sp_sent_type] +=1
        pred_sent_type_list.append(sp_sent_type)
        ###+++++++++
        sp_para_preds = list(set([_[0] for _ in sp_predictions]))
        para_type = set_comparison(prediction_list=sp_para_preds, true_list=sp_para_golds)
        prediction_para_type_counter[para_type] += 1
        pred_doc_type_list.append(para_type)
        ###+++++++++
        if raw_answer not in ['yes', 'no']:
            yes_no_span_true.append('span')
        else:
            yes_no_span_true.append(raw_answer)

        if ans_prediction not in ['yes', 'no']:
            yes_no_span_predictions.append('span')
        else:
            yes_no_span_predictions.append(ans_prediction)

        ans_type = 'em'
        if raw_answer not in ['yes', 'no']:
            if raw_answer == ans_prediction:
                ans_type = 'em'
            elif raw_answer in ans_prediction:
                # print('{}: {} |{}'.format(qid, raw_answer, ans_prediction))
                # print('-'*75)
                ans_type = 'super_of_gold'
            elif ans_prediction in raw_answer:
                # print('{}: {} |{}'.format(qid, raw_answer, ans_prediction))
                # print('-'*75)
                ans_type = 'sub_of_gold'
            else:
                ans_pred_tokens = ans_prediction.split(' ')
                ans_raw_tokens = raw_answer.split(' ')
                is_empty_set = len(set(ans_pred_tokens).intersection(set(ans_raw_tokens))) == 0
                if is_empty_set:
                    ans_type = 'no_over_lap'
                else:
                    ans_type = 'others'
        else:
            if raw_answer == ans_prediction:
                ans_type = 'em'
            else:
                ans_type = 'others'

        prediction_ans_type_counter[ans_type] += 1
        pred_ans_type_list.append(ans_type)


        # print('{} | {} | {}'.format(ans_type, raw_answer, ans_prediction))

    print(len(pred_sent_type_list), len(pred_ans_type_list), len(pred_doc_type_list))

    supp_sent_compare_type = ['equal', 'less', 'more']
    result_types = ['em', 'sub_of_gold', 'super_of_gold', 'no_over_lap', 'others']
    supp_sent_comp_dict = dict([(y, x) for x, y in enumerate(supp_sent_compare_type)])
    supp_sent_type_dict = dict([(y, x) for x, y in enumerate(result_types)])
    assert len(pred_sent_type_list) == len(pred_sent_count_list)
    print(len(pred_sent_type_list), len(pred_sent_count_list))
    conf_supp_sent_matrix = np.zeros((len(supp_sent_compare_type), len(result_types)), dtype=np.long)
    for idx in range(len(pred_sent_type_list)):
        comp_type_i = pred_sent_count_list[idx]
        supp_sent_type_i = pred_sent_type_list[idx]
        comp_idx_i = supp_sent_comp_dict[comp_type_i]
        supp_sent_idx_i = supp_sent_type_dict[supp_sent_type_i]
        conf_supp_sent_matrix[comp_idx_i][supp_sent_idx_i] += 1
    print('Sent Type vs Sent Count conf matrix:\n{}'.format(conf_supp_sent_matrix))
    print('Sum of matrix = {}'.format(conf_supp_sent_matrix.sum()))


    conf_matrix = confusion_matrix(yes_no_span_true, yes_no_span_predictions, labels=["yes", "no", "span"])
    conf_ans_sent_matrix = confusion_matrix(pred_sent_type_list, pred_ans_type_list, labels=result_types)
    print('*' * 75)
    print('Ans type conf matrix:\n{}'.format(conf_matrix))
    print('*' * 75)
    print('Sent vs ans conf matrix:\n{}'.format(conf_ans_sent_matrix))
    print('*' * 75)
    print("Ans prediction type: {}".format(prediction_ans_type_counter))
    print("Sent prediction type: {}".format(prediction_sent_type_counter))
    print("Para prediction type: {}".format(prediction_para_type_counter))
    print('*' * 75)

    conf_matrix_para_vs_sent = confusion_matrix(pred_doc_type_list, pred_sent_type_list, labels=result_types)
    print('Para Type vs Sent Type conf matrix:\n{}'.format(conf_matrix_para_vs_sent))
    print('*' * 75)
    conf_matrix_para_vs_ans = confusion_matrix(pred_doc_type_list, pred_ans_type_list, labels=result_types)
    print('Para Type vs ans Type conf matrix:\n{}'.format(conf_matrix_para_vs_ans))
    para_counter = Counter(pred_para_count_list)
    print('Para counter : {}'.format(para_counter))
    # pred_sent_para_type_counter = Counter()
    # for (sent_type, para_type) in zip(pred_doc_type_list, pred_sent_type_list):
    #     pred_sent_para_type_counter[(sent_type, para_type)] += 1
    # print('*' * 75)
    # for key, value in dict(pred_sent_para_type_counter).items():
    #     print('{} vs {}: {}'.format(key[0], key[1], value))
    # print('Para sent type: {}'.format(pred_sent_para_type_counter))


def error_analysis_question_type(raw_data, predictions, tokenizer, use_ent_ans=False):

    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_answer(prediction, gold):
        em = exact_match_score(prediction, gold)
        f1, prec, recall = f1_score(prediction, gold)
        return (em, prec, recall, f1)

    def update_sp(prediction, gold):
        cur_sp_pred = set(map(tuple, prediction))
        gold_sp_pred = set(map(tuple, gold))
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                tp += 1
            else:
                fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0
        return (em, prec, recall, f1)
    ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    type_metric_dict = {}
    for row in raw_data:
        question_type = row['type']
        if question_type not in type_metric_dict:
            type_metric_dict[question_type] = []

        qid = row['_id']
        sp_predictions = predictions['sp'][qid]
        sp_predictions = [(x[0], x[1]) for x in sp_predictions]
        sp_golds = row['supporting_facts']
        sp_golds = [(x[0], x[1]) for x in sp_golds]
        sp_metrics = update_sp(prediction=sp_predictions, gold=sp_golds)

        if qid == '5add114a5542994734353826':
            for x in row['context']:
                print('title ', x[0])
                for y_idx, y in enumerate(x[1]):
                    print('sentence', y_idx, y)

        ans_prediction = predictions['answer'][qid]
        raw_answer = row['answer']
        raw_answer = normalize_answer(raw_answer)
        ans_prediction = normalize_answer(ans_prediction)
        ans_metrics = update_answer(prediction=ans_prediction, gold=raw_answer)


        type_metric_dict[question_type].append((ans_metrics, sp_metrics))

    for key, value in type_metric_dict.items():
        print('question type = {}'.format(key))
        answer_em, answer_prec, answer_recall, answer_f1 = 0.0, 0.0, 0.0, 0.0
        sp_em, sp_prec, sp_recall, sp_f1 = 0.0, 0.0, 0.0, 0.0
        type_count = len(value)
        all_joint_em, all_joint_f1 = 0.0, 0.0
        for ans_tup, sp_tup in value:
            answer_em += ans_tup[0]
            answer_prec += ans_tup[1]
            answer_recall += ans_tup[2]
            answer_f1 += ans_tup[3]

            sp_em += sp_tup[0]
            sp_prec += sp_tup[1]
            sp_recall += sp_tup[2]
            sp_f1 += sp_tup[3]

            joint_prec = ans_tup[1] * sp_tup[1]
            joint_recall = ans_tup[2] * sp_tup[2]
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = ans_tup[0] * sp_tup[0]

            all_joint_f1 += joint_f1
            all_joint_em += joint_em

        print('ans {}\t{}\t{}\t{}'.format(answer_em/type_count, answer_recall/type_count, answer_prec/type_count, answer_f1/type_count))
        print('sup {}\t{}\t{}\t{}'.format(sp_em / type_count, sp_recall / type_count, sp_prec / type_count,
                                          sp_f1 / type_count))
        print('joint em ', all_joint_em/type_count)
        print('joint f1 ', all_joint_f1/type_count)


def prediction_score_analysis(raw_data, predictions, prediction_scores):
    def positive_neg_score(scores, mask, names, gold_names, pred_names):
        assert len(scores) == len(mask)
        print(type(scores), type(mask))
        return

    for row in raw_data:
        qid = row['_id']
        sp_predictions = predictions['sp'][qid]
        sp_predictions = [(x[0], x[1]) for x in sp_predictions]
        sp_para_predictions = list(set([x[0] for x in sp_predictions]))

        sp_golds = row['supporting_facts']
        sp_golds = [(x[0], x[1]) for x in sp_golds]
        sp_para_golds = list(set([_[0] for _ in sp_golds]))

        res_scores = prediction_scores[qid]
        sp_scores = res_scores['sp_score']
        sp_mask = res_scores['sp_mask']
        sp_names = res_scores['sp_names']
        sp_names = [(x[0], x[1]) for x in sp_names]
        positive_neg_score(scores=sp_scores, mask=sp_mask, names=sp_names, gold_names=sp_golds, pred_names=sp_predictions)

        # for key, value in sp_scores.items():
        #     print(key, value)