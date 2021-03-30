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
        support_sent_mask = batch
        ####################################################################

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]
            ##+++++++++++++++++++++++++
            sent_names_i = example_dict[cur_id].sent_names
            sent_scores_i = predict_support_np[i]
            ##+++++++++++++++++++++++++

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