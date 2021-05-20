import json
import numpy as np
from tqdm import tqdm
import torch
import shutil
import os
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from csr_mhqa.utils import convert_to_tokens
from utils.gpu_utils import single_free_cuda
import torch.nn.functional as F
from leaderboardscripts.lb_postprocess_utils import row_x_feat_extraction, np_sigmoid
from leaderboardscripts.lb_postprocess_utils import RangeDataset
from torch.utils.data import DataLoader

def jd_unified_test_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, threshold=0.45, dev_gold_file=None,
                          output_score_file=None):
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}
    sp_dict = {}

    ##++++++
    prediction_res_score_dict = {}
    ##++++++

    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            start, end, q_type, paras, sent, ent, yp1, yp2, cls_emb = model(batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        predict_support_logit_np = sent[:, :, 1].data.cpu().numpy()
        support_sent_mask_np = batch['sent_mask'].data.cpu().numpy()
        predict_support_para_logit_np = paras[:, :, 1].data.cpu().numpy()
        support_para_mask_np = batch['para_mask'].data.cpu().numpy()

        cls_emb_np = cls_emb.data.cpu().numpy()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i in range(predict_support_np.shape[0]):
            cur_id = batch['ids'][i]
            sp_dict[cur_id] = []
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_pred_ = {'sp_score': predict_support_logit_np[i].tolist(), 'sp_mask': support_sent_mask_np[i].tolist(),
                          'sp_names': example_dict[cur_id].sent_names}
            para_pred_ = {'sp_para_score': predict_support_para_logit_np[i].tolist(), 'sp_para_mask': support_para_mask_np[i].tolist(),
                          'sp_para_names': example_dict[cur_id].para_names}

            cls_emb_ = {'cls_emb': cls_emb_np[i].tolist()}
            res_score = {**sent_pred_, **cls_emb_, **para_pred_}
            prediction_res_score_dict[cur_id] = res_score
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if predict_support_np[i, j] > threshold:
                    sp_dict[cur_id].append(example_dict[cur_id].sent_names[j])

    prediction = {'answer': answer_dict,
                  'sp': sp_dict,
                  'type': answer_type_dict,
                  'type_prob': answer_type_prob_dict}
    tmp_file = os.path.join(os.path.dirname(prediction_file), 'test_prediction.json')
    with open(tmp_file, 'w') as f:
        json.dump(prediction, f)
    metrics = hotpot_eval(tmp_file, dev_gold_file)
    json.dump(metrics, open(eval_file, 'w'))

    if output_score_file is not None:
        with open(output_score_file, 'w') as f:
            json.dump(prediction_res_score_dict, f)
        print('Saving {} score records into {}'.format(len(prediction_res_score_dict), output_score_file))
    return metrics

def jd_post_process_feature_extraction(raw_file_name, score_file_name, feat_file_name):
    with open(raw_file_name, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)
    with open(score_file_name, 'r', encoding='utf-8') as reader:
        score_data = json.load(reader)
    feat_dict = {}
    for case in tqdm(raw_data):
        key = case['_id']
        if key in score_data:
            score_case = score_data[key]
            x_feat = row_x_feat_extraction(row=score_case)
            feat_dict[key] = {'x_feat': x_feat}
    json.dump(feat_dict, open(feat_file_name, 'w'))
    print('Saving {} records into {}'.format(len(feat_dict), feat_file_name))

def jd_postprocess_score_prediction(args, model, data_loader, threshold_category):
    alpha = args.alpha
    pred_score_dict = {}
    for batch_idx, batch in tqdm(enumerate(data_loader)):
        # ++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'id'}:
                batch[key] = value.to(args.device)
        # ++++++++++++++++++++
        with torch.no_grad():
            start_scores, end_scores, y1, y2 = model(batch['x_feat'], return_yp=True)
            start_indexes = y1.data.cpu().numpy()
            end_indexes = y2.data.cpu().numpy()
        for i in range(start_indexes.shape[0]):
            key = batch['id'][i]
            start_i = int(start_indexes[i])
            end_i = int(end_indexes[i])
            if start_i > end_i:
                print('here')
            pred_idx_i = (start_i + end_i) // 2 + 1 ## better for EM
            if pred_idx_i == len(threshold_category):
                print('hhhhhhhhh')
            score_i = (threshold_category[start_i][1] * (1 - alpha) + threshold_category[end_i][0] * alpha) ## better for F1
            score_i = (threshold_category[pred_idx_i][1] + score_i)/2
            pred_score_dict[key] = score_i
    return pred_score_dict

def jd_adaptive_threshold_prediction(args, model, feat_dict_file_name):
    if torch.cuda.is_available():
        device_ids, _ = single_free_cuda()
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')
    data_feat = RangeDataset(json_file_name=feat_dict_file_name)
    data_loader = DataLoader(dataset=data_feat,
                                 shuffle=False,
                                 collate_fn=RangeDataset.collate_fn,
                                 batch_size=args.test_batch_size)
    model.to(device)
    model.eval()
    pred_score_dict = {}
    total_count = 0
    for batch in data_loader:
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in ['id']:
                batch[key] = value.to(device)
        with torch.no_grad():
            scores = model(batch['x_feat'])
            scores = scores.squeeze(-1)
            scores = torch.sigmoid(scores)
            score_np = scores.data.cpu().numpy()
            for i in range(score_np.shape[0]):
                key = batch['id'][i]
                total_count = total_count + 1
                score_i = score_np[i]
                pred_score_dict[key] = float(score_i)
    return pred_score_dict

def jd_adaptive_threshold_post_process(full_file, prediction_answer_file, score_dict_file, threshold_pred_dict_file):
    with open(prediction_answer_file, 'r', encoding='utf-8') as reader:
        pred_data = json.load(reader)
    print('Loading {} records from {}'.format(len(pred_data), prediction_answer_file))

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)
    print('Loading {} records from {}'.format(len(full_data), full_file))

    with open(score_dict_file, 'r', encoding='utf-8') as reader:
        score_dict = json.load(reader)
    print('Loading {} records from {}'.format(len(score_dict), score_dict_file))

    with open(threshold_pred_dict_file, 'r', encoding='utf-8') as reader:
        threshold_pred_dict = json.load(reader)
    print('Loading {} records from {}'.format(len(threshold_pred_dict), threshold_pred_dict_file))

    def inner_supp_fact(score_case, threshold_case):
        sp_pred_scores = score_case['sp_score']
        sp_pred_mask = score_case['sp_mask']
        assert len(sp_pred_scores) == len(sp_pred_mask)
        sent_num = int(sum(sp_pred_mask))
        sp_names = score_case['sp_names']
        assert sent_num <= len(sp_names)
        sp_pred_scores = np_sigmoid(np.array(sp_pred_scores)[:sent_num])
        sorted_idx = np.argsort(sp_pred_scores)[::-1]


        pred_supp_fact_res = []
        pred_supp_fact_ids = []
        supp_para_names = {}
        for i in range(sent_num):
            if sp_pred_scores[i] >= threshold_case:
                pred_supp_fact_res.append(sp_names[i])
                pred_supp_fact_ids.append(i)
                if sp_names[i][0] not in supp_para_names:
                    supp_para_names[sp_names[i][0]] = 1
                else:
                    supp_para_names[sp_names[i][0]] = supp_para_names[sp_names[i][0]] + 1
        supp_para_names = set([x[0] for x in pred_supp_fact_res])
        return pred_supp_fact_res, len(supp_para_names) != 2

    pred_answer_dict = pred_data['answer']
    pred_type_dict = pred_data['type']
    pred_supp_fact_dict = {}
    para_count = 0
    for case in tqdm(full_data):
        key = case['_id']
        score_case = score_dict[key]
        threshold_case = threshold_pred_dict[key]
        supp_fact_i, flag = inner_supp_fact(score_case=score_case, threshold_case=threshold_case)
        pred_supp_fact_dict[key] = supp_fact_i
        if flag:
            para_count = para_count + 1
        # print(threshold_case)
        # print(score_case['sp_score'])
    prediction_res = {'answer': pred_answer_dict,
                  'sp': pred_supp_fact_dict,
                  'type': pred_type_dict}
    print(para_count)
    return prediction_res


def jd_unified_eval_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file=None):
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    thresholds = np.arange(0.1, 1.0, 0.025)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]

    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            start, end, q_type, paras, sent, ent, yp1, yp2, _ = model(batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)

        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break

                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i]:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []

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

    return best_metrics, best_threshold

def convert_answer_to_sent_names(examples, features, batch, y1, y2, q_type_prob):
    answer2sent_name_dict = {}
    support_sent_mask_np = batch['sent_mask'].data.cpu().numpy()
    ids = batch['ids']
    #+++++++++++++
    answer_dict, answer_type_dict = {}, {}
    answer_type_prob_dict = {}
    q_type = np.argmax(q_type_prob, 1)
    # +++++++++++++
    def get_ans_from_pos(qid, y1, y2):
        feature = features[qid]
        example = examples[qid]
        tok_to_orig_map = feature.token_to_orig_map
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
        ###++++++++++++++++++++++++++++++
        assert support_sent_mask_np[i].sum() == len(feature.__dict__['sent_spans'])
        sent_spans = feature.sent_spans
        para_spans = feature.para_spans
        ###++++++++++++++++++++++++++++++
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
        answer2sent_name_dict[qid] = answer_sent_name
    return answer_dict, answer_type_dict, answer2sent_name_dict


def post_process_sent_para(cur_id, example_dict, feature_dict, sent_scores_np_i, sent_mask_np_i, para_scores_np_i, para_mask_np_i):
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
    total_para_num_i = len(para_names_i)
    para_mask_i = para_mask_np_i
    assert total_para_num_i >= para_mask_i.sum()
    para_mask_num = int(para_mask_i.sum())

    sorted_idxes = np.argsort(sent_scores_i)[::-1]
    topk_sent_idxes = sorted_idxes[:2].tolist()
    topk_sent_selected_paras = set([sent_names_i[_][0] for _ in topk_sent_idxes])
    if len(topk_sent_selected_paras) < 2 and para_mask_num > 1:
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
    para_scores_i[para_mask_i == 0] = -100
    para_sorted_idxes = np.argsort(para_scores_i)[::-1]
    if para_mask_num == 1:
        topk_para_idxes = para_sorted_idxes[:1]
    else:
        topk_para_idxes = para_sorted_idxes[:2]
    topk_pred_paras = set([para_names_i[_] for _ in topk_para_idxes])
    assert len(topk_pred_paras) <= para_mask_num
    diff_para = topk_pred_paras.difference(topk_sent_selected_paras)
    #++++++++++++++++++++++++++
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
            # assert sorted_idx_i >=0
            if sorted_idx_i < 0:
                print(cur_id)
                print(feature_dict[cur_id].para_spans)
                print(feature_dict[cur_id].sent_spans)
                print(topk_pred_paras)
                print(topk_sent_selected_paras)
                print(para_mask_i)
                print(sent_mask_i)
                print(para)
                print(topk)
                print(sent_mask_num)
                print(para_mask_num)
                print(sent_names_i)
            # assert sorted_idx_i >= 0
            if sorted_idx_i >=0:
                diff_para_sent_idxes.append(sorted_idx_i)
        diff_para_sent_names = [sent_names_i[_] for _ in diff_para_sent_idxes]
        if len(diff_para) != len(diff_para_sent_names):
            print(diff_para)
            print(diff_para_sent_names)
            print(sent_names_i)
            print(topk_sent_selected_paras)
        # assert len(diff_para_sent_names) == len(diff_para)
    # ++++++++++++++++++++++++++
    return topk_score_ref, cut_sent_flag, topk_pred_sent_names, diff_para_sent_names, topk_pred_paras


def post_process_technique(cur_sp_pred, topk_pred_sent_names, diff_para_sent_names, topk_pred_paras, ans_sent_name):
    if len(cur_sp_pred) < 2:
        post_process_sp_pred = topk_pred_sent_names
    else:
        post_process_sp_pred = cur_sp_pred
    post_process_sp_pred = [x for x in post_process_sp_pred if x[0] in topk_pred_paras]
    number_of_paras = len(set([x[0] for x in post_process_sp_pred]))
    if number_of_paras == 1 and len(diff_para_sent_names) > 0:
        post_process_sp_pred.extend(diff_para_sent_names)
    if (ans_sent_name is not None) and (ans_sent_name not in post_process_sp_pred):
        post_process_sp_pred.append(ans_sent_name)
    return post_process_sp_pred


def jd_postprocess_unified_eval_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file=None):
    model.eval()
    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    thresholds = np.arange(0.1, 1.0, 0.025)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]

    for batch in tqdm(dataloader):
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            start, end, q_type, paras, sent, ent, yp1, yp2, cls_emb = model(batch, return_yp=True)
        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        _, _, answer_sent_name_dict_ = convert_answer_to_sent_names(example_dict, feature_dict, batch,
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
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
                post_process_sent_para(cur_id=cur_id, example_dict=example_dict, feature_dict=feature_dict,
                                       sent_scores_np_i=predict_support_np[i], sent_mask_np_i=support_sent_mask_np[i],
                                       para_scores_np_i=predict_support_para_np[i], para_mask_np_i=support_para_mask_np[i])
            ans_sent_name = answer_sent_name_dict_[cur_id]
            ##+++++++++++++++++++++++++
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i] * topk_score_ref:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []
                ##+++++
                post_process_thresh_i_sp_pred = post_process_technique(cur_sp_pred=cur_sp_pred[thresh_i],
                                                               topk_pred_paras=topk_pred_paras,
                                                               topk_pred_sent_names=topk_pred_sent_names,
                                                               diff_para_sent_names=diff_para_sent_names,
                                                               ans_sent_name=ans_sent_name)
                total_sp_dict[thresh_i][cur_id].extend(post_process_thresh_i_sp_pred)
                # # +++++++++++++++++++++++++++

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
    return best_metrics, best_threshold


def jd_postprecess_unified_test_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, threshold=0.45, dev_gold_file=None,
                          output_score_file=None):
    model.eval()
    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}
    sp_dict = {}
    ##++++++
    prediction_res_score_dict = {}
    ##++++++
    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            start, end, q_type, paras, sent, ent, yp1, yp2, cls_emb = model(batch, return_yp=True)
        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        _, _, answer_sent_name_dict_ = convert_answer_to_sent_names(example_dict, feature_dict, batch,
                                                                    yp1.data.cpu().numpy().tolist(),
                                                                    yp2.data.cpu().numpy().tolist(),
                                                                    type_prob)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)
        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()
        predict_support_para_np = torch.sigmoid(paras[:, :, 1]).data.cpu().numpy()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        predict_support_logit_np = sent[:, :, 1].data.cpu().numpy()
        support_sent_mask_np = batch['sent_mask'].data.cpu().numpy()
        predict_support_para_logit_np = paras[:, :, 1].data.cpu().numpy()
        support_para_mask_np = batch['para_mask'].data.cpu().numpy()
        cls_emb_np = cls_emb.data.cpu().numpy()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i in range(predict_support_np.shape[0]):
            cur_id = batch['ids'][i]
            cur_sp_pred = []
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_pred_ = {'sp_score': predict_support_logit_np[i].tolist(), 'sp_mask': support_sent_mask_np[i].tolist(),
                          'sp_names': example_dict[cur_id].sent_names}
            para_pred_ = {'sp_para_score': predict_support_para_logit_np[i].tolist(), 'sp_para_mask': support_para_mask_np[i].tolist(),
                          'sp_para_names': example_dict[cur_id].para_names}

            cls_emb_ = {'cls_emb': cls_emb_np[i].tolist()}
            res_score = {**sent_pred_, **cls_emb_, **para_pred_}
            prediction_res_score_dict[cur_id] = res_score
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            topk_score_ref, cut_sent_flag, topk_pred_sent_names, diff_para_sent_names, topk_pred_paras = \
                post_process_sent_para(cur_id=cur_id, example_dict=example_dict, feature_dict=feature_dict,
                                       sent_scores_np_i=predict_support_np[i], sent_mask_np_i=support_sent_mask_np[i],
                                       para_scores_np_i=predict_support_para_np[i], para_mask_np_i=support_para_mask_np[i])
            ans_sent_name = answer_sent_name_dict_[cur_id]

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if predict_support_np[i, j] > topk_score_ref * threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])

            post_process_sp_pred = post_process_technique(cur_sp_pred=cur_sp_pred,
                                                                   topk_pred_paras=topk_pred_paras,
                                                                   topk_pred_sent_names=topk_pred_sent_names,
                                                                   diff_para_sent_names=diff_para_sent_names,
                                                                   ans_sent_name=ans_sent_name)
            sp_dict[cur_id] = post_process_sp_pred

    prediction = {'answer': answer_dict,
                  'sp': sp_dict,
                  'type': answer_type_dict,
                  'type_prob': answer_type_prob_dict}
    if output_score_file is not None:
        with open(output_score_file, 'w') as f:
            json.dump(prediction_res_score_dict, f)
        print('Saving {} score records into {}'.format(len(prediction_res_score_dict), output_score_file))
    return prediction