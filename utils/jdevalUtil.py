import json
import numpy as np
from tqdm import tqdm
import torch
import shutil
import os
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from csr_mhqa.utils import convert_to_tokens
import torch.nn.functional as F

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
            sent_names_i = example_dict[cur_id].sent_names
            total_sent_num_i = len(sent_names_i)
            sent_scores_i = predict_support_np[i]
            sent_mask_i = support_sent_mask_np[i]
            sent_scores_i[sent_mask_i == 0] = -100
            # print(sent_scores_i)
            print('+' * 100)
            if total_sent_num_i != sent_mask_i.sum():
                cut_sentence_count = cut_sentence_count + 1
                # print(sent_names_i)
                # print(total_sent_num_i)
                # print(sent_mask_i.sum())
            assert total_sent_num_i >= sent_mask_i.sum()
            # for temp_i in range(total_sent_num_i):
            #     print('{}\t{:.4f}'.format(sent_names_i[temp_i], sent_scores_i[temp_i]))
            sorted_idxes = np.argsort(sent_scores_i)[::-1]
            topk_sent_idxes = sorted_idxes[:2]
            topk_pred_sents = [sent_names_i[_] for _ in topk_sent_idxes]

            para_names_i = example_dict[cur_id].para_names
            para_scores_i = predict_support_para_np[i]
            para_mask_i = support_para_mask_np[i]
            para_scores_i[para_mask_i == 0] = -100
            para_sorted_idxes = np.argsort(para_scores_i)[::-1]
            topk_para_idxes = para_sorted_idxes[:2]
            topk_pred_paras = set([para_names_i[_] for _ in topk_para_idxes])
            top2_para_total_sent_num_i = len([x for x in sent_names_i if x[0] in topk_pred_paras])
            ##+++++++++++++++++++++++++

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break

                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i]:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])
                    # # +++++++++++++++++++++++++++
                    # temp = [x for x in cur_sp_pred[thresh_i] if x[0] in topk_pred_paras]
                    # cur_sp_pred[thresh_i] = temp
                    # if len(cur_sp_pred[thresh_i]) < 2:
                    #     cur_sp_pred[thresh_i].extend(topk_pred_sents)
                    # # if len(cur_sp_pred[thresh_i]) >= top2_para_total_sent_num_i and len(cur_sp_pred[thresh_i]) >=4:
                    # # +++++++++++++++++++++++++++

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

    print('Number of examples with cutted sentences = {}'.format(cut_sentence_count))
    return best_metrics, best_threshold