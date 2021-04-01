import json
import numpy as np
from tqdm import tqdm
import torch
import shutil
import sys
import os
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from csr_mhqa.utils import convert_to_tokens
import torch.nn.functional as F
from eval.hotpot_evaluate_v1 import exact_match_score

def recall_computation(prediction, gold):
    gold_set = set(gold)
    gold_count = len(gold_set)
    tp = 0
    prediction_set = set(prediction)
    prediction = list(prediction_set)
    for pred in prediction:
        if pred in gold_set:
            tp = tp + 1
    recall = 1.0 * tp /gold_count
    em_recall = 1.0 if recall == 1.0 else 0.0
    return em_recall

def doc_recall_eval(doc_prediction, gold_file):
    with open(gold_file) as f:
        gold = json.load(f)
    recall_list = []
    for dp in gold:
        cur_id = dp['_id']
        support_facts = dp['supporting_facts']
        support_doc_titles = list(set([_[0] for _ in support_facts]))
        pred_doc_titles = doc_prediction[cur_id]
        em_recall = recall_computation(prediction=pred_doc_titles, gold=support_doc_titles)
        recall_list.append(em_recall)
    em_recall = sum(recall_list)/len(recall_list)
    return em_recall

def jd_eval_model(args, encoder, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file):
    encoder.eval()
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    # dataloader.refresh()
    #++++++
    cut_sentence_count = 0
    #++++++

    thresholds = np.arange(0.1, 1.0, 0.05)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]

    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            inputs = {'input_ids':      batch['context_idxs'],
                      'attention_mask': batch['context_mask'],
                      'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet', 'electra'] else None}  # XLM don't use segment_ids
            outputs = encoder(**inputs)
            ####++++++++++++++++++++++++++++++++++++++
            if args.model_type == 'electra':
                batch['context_encoding'] = outputs.last_hidden_state
            else:
                batch['context_encoding'] = outputs[0]
            ####++++++++++++++++++++++++++++++++++++++
            batch['context_mask'] = batch['context_mask'].float().to(args.device)
            start, end, q_type, paras, sent, ent, yp1, yp2 = model(batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print('ent_prediction', ent.shape)
        # print('ent_mask', batch['ans_cand_mask'])
        # print('gold_ent', batch['is_gold_ent'])
        ent_pre_prob = torch.sigmoid(ent).data.cpu().numpy()
        _, _, answer_sent_name_dict_ = convert_answer_to_sent_names(example_dict, feature_dict, batch,
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob, ent_pre_prob)
        # sys.exit(0)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()
        ####################################################################
        support_sent_mask_np = batch['sent_mask'].data.cpu().numpy()
        predict_support_para_np = torch.sigmoid(paras[:, :, 1]).data.cpu().numpy()
        support_para_mask_np = batch['para_mask'].data.cpu().numpy()
        ####################################################################

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]
            ##+++++++++++++++++++++++++
            topk_score_ref, cut_sent_flag, topk_pred_sent_names, diff_para_sent_names, topk_pred_paras = \
                post_process_sent_para(cur_id=cur_id, example_dict=example_dict,
                                       sent_scores_np_i=predict_support_np[i], sent_mask_np_i=support_sent_mask_np[i],
                                       para_scores_np_i=predict_support_para_np[i], para_mask_np_i=support_para_mask_np[i])
            ans_sent_name = answer_sent_name_dict_[cur_id]
            ##+++++++++++++++++++++++++

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break

                for thresh_i in range(N_thresh):
                    # if predict_support_np[i, j] > thresholds[thresh_i]:
                    if predict_support_np[i, j] > thresholds[thresh_i] * topk_score_ref:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []
                ##+++++
                # +++++++++++++++++++++++++++
                post_process_thresh_i_sp_pred = post_process_technique(cur_sp_pred=cur_sp_pred[thresh_i],
                                                               topk_pred_paras=topk_pred_paras,
                                                               topk_pred_sent_names=topk_pred_sent_names,
                                                               diff_para_sent_names=diff_para_sent_names,
                                                               ans_sent_name=ans_sent_name)
                # print('former ', cur_sp_pred[thresh_i])
                # print('post ', post_process_thresh_i_sp_pred)
                # # +++++++++++++++++++++++++++
                ##+++++
                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold

    best_metrics, best_threshold = choose_best_threshold(answer_dict, prediction_file)
    json.dump(best_metrics, open(eval_file, 'w'))

    print('Number of examples with cutted sentences = {}'.format(cut_sentence_count))
    return best_metrics, best_threshold

def post_process_sent_para(cur_id, example_dict, sent_scores_np_i, sent_mask_np_i, para_scores_np_i, para_mask_np_i):
    sent_names_i = example_dict[cur_id].sent_names
    para_names_i = example_dict[cur_id].para_names
    cut_sent_flag = False
    total_sent_num_i = len(sent_names_i)
    sent_scores_i = sent_scores_np_i
    sent_mask_i = sent_mask_np_i
    sent_scores_i[sent_mask_i == 0] = -100
    if total_sent_num_i != sent_mask_i.sum():
        cut_sent_flag = True
    assert total_sent_num_i >= sent_mask_i.sum()
    sent_mask_num = int(sent_mask_i.sum())

    sorted_idxes = np.argsort(sent_scores_i)[::-1]
    topk_sent_idxes = sorted_idxes[:2].tolist()
    topk_sent_selected_paras = set([sent_names_i[_][0] for _ in topk_sent_idxes])
    if len(topk_sent_selected_paras) < 2:
        for s_idx in range(2, sent_mask_num):
            topk_sent_idxes.append(sorted_idxes[s_idx])
            if sent_names_i[sorted_idxes[s_idx]][0] not in topk_sent_selected_paras:
                break
            else:
                continue
    topk_score_ref = sent_scores_i[topk_sent_idxes[-1]]
    topk_pred_sent_names = [sent_names_i[_] for _ in topk_sent_idxes]
    topk_sent_selected_paras = set([_[0] for _ in topk_pred_sent_names])

    para_scores_i = para_scores_np_i
    para_mask_i = para_mask_np_i
    para_scores_i[para_mask_i == 0] = -100
    para_sorted_idxes = np.argsort(para_scores_i)[::-1]
    topk_para_idxes = para_sorted_idxes[:2]
    topk_pred_paras = set([para_names_i[_] for _ in topk_para_idxes])
    assert len(topk_pred_paras) >= 2
    diff_para = topk_pred_paras.difference(topk_sent_selected_paras)
    #++++++++
    diff_para_sent_idxes = []
    diff_para_sent_names = []
    def find_largest_sent_idx(para, topk, sent_mask_num, sent_names):
        for s_idx_i in range(topk, sent_mask_num):
            sorted_idx_i = sorted_idxes[s_idx_i]
            if sent_names[sorted_idx_i][0] == para:
                return sorted_idx_i
        return -1

    if len(diff_para) > 0:
        topk = len(topk_sent_idxes)
        for para in list(diff_para):
            sorted_idx_i = find_largest_sent_idx(para=para, topk=topk, sent_mask_num=sent_mask_num,
                                                 sent_names=sent_names_i)
            if sorted_idx_i >= 0:
                diff_para_sent_idxes.append(sorted_idx_i)
            else:
                print(sent_names_i)
                print(para)
            # for s_idx_i in range(topk, sent_mask_num):
            #     sorted_idx_i = sorted_idxes[s_idx_i]
            #     if sent_names_i[sorted_idx_i][0] == para:
            #         diff_para_sent_idxes.append(sorted_idx_i)
            #         break
        diff_para_sent_names = [sent_names_i[_] for _ in diff_para_sent_idxes]
        if len(diff_para) != len(diff_para_sent_names):
            print(diff_para)
            print(diff_para_sent_names)
            print(sent_names_i)
            assert len(diff_para_sent_names) == len(diff_para)
    # ++++++++
    return topk_score_ref, cut_sent_flag, topk_pred_sent_names, diff_para_sent_names, topk_pred_paras

def post_process_technique(cur_sp_pred, topk_pred_sent_names, diff_para_sent_names, topk_pred_paras, ans_sent_name):
    if len(cur_sp_pred) < 2:
        post_process_sp_pred = topk_pred_sent_names
    else:
        post_process_sp_pred = cur_sp_pred
    post_process_sp_pred = [x for x in post_process_sp_pred if x[0] in topk_pred_paras]
    number_of_paras = len(set([x[0] for x in post_process_sp_pred]))
    if number_of_paras == 1:
        assert len(diff_para_sent_names) > 0
        post_process_sp_pred.extend(diff_para_sent_names)
    if (ans_sent_name is not None) and (ans_sent_name not in post_process_sp_pred):
        post_process_sp_pred.append(ans_sent_name)
    # print(len(set([x[0] for x in post_process_sp_pred])))
    # if len(cur_sp_pred) != len(post_process_sp_pred):
    #     print('cur', cur_sp_pred)
    #     print('post', post_process_sp_pred)
    # print('*' * 80)
    return post_process_sp_pred

def convert_answer_to_sent_names(examples, features, batch, y1, y2, q_type_prob, ent_pred_prob):
    answer2sent_name_dict = {}
    support_sent_mask_np = batch['sent_mask'].data.cpu().numpy()
    ids = batch['ids']
    support_para_mask_np = batch['para_mask'].data.cpu().numpy()
    ans_cand_mask = batch['ans_cand_mask'].data.cpu().numpy()
    ent_mask = batch['ent_mask'].data.cpu().numpy()
    is_gold_ent = batch['is_gold_ent'].data.cpu().numpy()
    #+++++++++++++
    answer_dict, answer_type_dict = {}, {}
    answer_type_prob_dict = {}

    # print('is_gold_ent', batch['is_gold_ent'])

    q_type = np.argmax(q_type_prob, 1)
    # print(q_type)
    # +++++++++++++
    ent_prediction = np.argmax(ent_pred_prob, 1)

    def get_ans_from_pos(qid, y1, y2):
        feature = features[qid]
        example = examples[qid]

        tok_to_orig_map = feature.token_to_orig_map
        orig_all_tokens = example.question_tokens + example.doc_tokens

        final_text = " "
        if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
            orig_tok_start = tok_to_orig_map[y1]
            orig_tok_end = tok_to_orig_map[y2]

            ques_tok_len = len(example.question_tokens)
            # print('question tokens = {}'.format(example.question_tokens))
            if orig_tok_start < ques_tok_len and orig_tok_end < ques_tok_len:
                ques_start_idx = example.question_word_to_char_idx[orig_tok_start]
                ques_end_idx = example.question_word_to_char_idx[orig_tok_end] + len(
                    example.question_tokens[orig_tok_end])
                final_text = example.question_text[ques_start_idx:ques_end_idx]
            else:
                orig_tok_start -= len(example.question_tokens)
                orig_tok_end -= len(example.question_tokens)
                ctx_start_idx = example.ctx_word_to_char_idx[orig_tok_start]
                ctx_end_idx = example.ctx_word_to_char_idx[orig_tok_end] + len(example.doc_tokens[orig_tok_end])
                # final_text = example.ctx_text[example.ctx_word_to_char_idx[orig_tok_start]:example.ctx_word_to_char_idx[
                #                                                                                orig_tok_end] + len(
                #     example.doc_tokens[orig_tok_end])]
                final_text = example.ctx_text[ctx_start_idx:ctx_end_idx]

        return final_text

    def get_sent_name_accord_ans(y1, y2, sent_spans, para_spans):
        ans_para_idx = -1
        ans_sent_idx = -1
        for para_idx, para_span in enumerate(para_spans):
            para_start_idx, para_end_idx, para_name = para_span
            if y1 >= para_start_idx and y2 <= para_end_idx:
                ans_para_idx = para_idx
                break
        if ans_para_idx >= 0:
            para_start_idx, para_end_idx, para_name = para_spans[ans_para_idx]
            sent_spans_filtered = [_ for _ in sent_spans if (_[0]>= para_start_idx and _[1] <= para_end_idx)]
            for sent_idx, sent_span in enumerate(sent_spans_filtered):
                sent_start_idx, sent_end_idx = sent_span
                if y1 >= sent_start_idx and y2 <= sent_end_idx:
                    ans_sent_idx = sent_idx
        if ans_para_idx >= 0 and ans_sent_idx >= 0:
            sent_name = (para_spans[ans_para_idx][2], ans_sent_idx)
        else:
            sent_name = None
        return sent_name

    for i, qid in enumerate(ids):
        feature = features[qid]
        example = examples[qid]
        ###++++++++++++++++++++++++++++++
        assert support_sent_mask_np[i].sum() == len(feature.__dict__['sent_spans'])
        sent_spans = feature.sent_spans
        para_spans = feature.para_spans
        entity_spans = feature.entity_spans
        # print('sent_spans', sent_spans)
        # print('para_spans', para_spans)
        ###++++++++++++++++++++++++++++++
        answer_text = ''
        answer_sent_name = None
        if q_type[i] in [0, 3]:
            answer_text = get_ans_from_pos(qid, y1[i], y2[i])
            answer_sent_name = get_sent_name_accord_ans(y1=y1[i], y2=y2[i], sent_spans=sent_spans, para_spans=para_spans)
        elif q_type[i] == 1:
            answer_text = 'yes'
        elif q_type[i] == 2:
            answer_text = 'no'
        else:
            raise ValueError("question type error")

        answer_dict[qid] = answer_text
        answer_type_prob_dict[qid] = q_type_prob[i].tolist()
        answer_type_dict[qid] = q_type[i].item()

        ###++++++++++++++++++++++++++++++
        # print('answer_sent_name', answer_sent_name, answer_text)
        answer2sent_name_dict[qid] = answer_sent_name
    return answer_dict, answer_type_dict, answer2sent_name_dict