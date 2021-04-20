from eval.hotpot_evaluate_v1 import update_answer, update_sp
import json
import numpy as np
import torch
import os
from tqdm import tqdm
from utils.jdevalUtil import post_process_sent_para, post_process_technique, convert_answer_to_sent_names
from csr_mhqa.utils import convert_to_tokens
import torch.nn.functional as F
import shutil

def train_eval(prediction_file, gold_file, train_type: str):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    total = 0
    for dp in gold:
        cur_id = dp['_id']
        train_cur_id = cur_id + '_' + train_type
        can_eval_joint = True
        ##+++++++++
        if train_cur_id not in prediction['answer']:
            #print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][train_cur_id], dp['answer'])
        if train_cur_id not in prediction['sp']:
            #print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][train_cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

            total += 1

    for k in metrics.keys():
        metrics[k] /= total

    return metrics


def jd_train_eval_model(args, encoder, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, train_gold_file, train_type, output_score_file=None):
    encoder.eval()
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}
    ##++++++
    prediction_res_score_dict = {}
    ##++++++
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
            start, end, q_type, paras, sent, ent, yp1, yp2, cls_emb = model(batch, return_yp=True, return_cls=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # print('ent_prediction', ent.shape)
        # print('ent_mask', batch['ans_cand_mask'])
        # print('gold_ent', batch['is_gold_ent'])
        ent_pre_prob = torch.sigmoid(ent).data.cpu().numpy()
        ent_mask_np = batch['ent_mask'].data.cpu().numpy()
        ans_cand_mask_np = batch['ans_cand_mask'].data.cpu().numpy()
        is_gold_ent_np = batch['is_gold_ent'].data.cpu().numpy()

        _, _, answer_sent_name_dict_ = convert_answer_to_sent_names(example_dict, feature_dict, batch,
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob, ent_pre_prob)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()
        ####################################################################
        support_sent_mask_np = batch['sent_mask'].data.cpu().numpy()
        predict_support_para_np = torch.sigmoid(paras[:, :, 1]).data.cpu().numpy()
        support_para_mask_np = batch['para_mask'].data.cpu().numpy()
        cls_emb_np = cls_emb.data.cpu().numpy()
        ####################################################################
        predict_support_logit_np = sent[:, :, 1].data.cpu().numpy()
        predict_support_para_logit_np = paras[:, :, 1].data.cpu().numpy()
        ent_pre_logit_np = ent.data.cpu().numpy()
        ####################################################################

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]
            ##+++++++++++++++++++++++++
            orig_supp_fact_id = example_dict[cur_id].sup_fact_id
            prune_supp_fact_id = feature_dict[cur_id].sup_fact_ids
            print('origi supp fact id {}'.format(orig_supp_fact_id))
            print('prune supp fact id {}'.format(prune_supp_fact_id))
            ##+++++++++++++++++++++++++
            topk_score_ref, cut_sent_flag, topk_pred_sent_names, diff_para_sent_names, topk_pred_paras = \
                post_process_sent_para(cur_id=cur_id, example_dict=example_dict, feature_dict=feature_dict,
                                       sent_scores_np_i=predict_support_np[i], sent_mask_np_i=support_sent_mask_np[i],
                                       para_scores_np_i=predict_support_para_np[i], para_mask_np_i=support_para_mask_np[i])
            ans_sent_name = answer_sent_name_dict_[cur_id]
            if cut_sent_flag:
                cut_sentence_count += 1
            ##+++++++++++++++++++++++++
            # sent_pred_ = {'sp_score': predict_support_np[i].tolist(), 'sp_mask': support_sent_mask_np[i].tolist(), 'sp_names': example_dict[cur_id].sent_names}
            # para_pred_ = {'para_score': predict_support_para_np[i].tolist(), 'para_mask': support_para_mask_np[i].tolist(), 'para_names': example_dict[cur_id].para_names}
            # ans_pred_ = {'ans_type': type_prob[i].tolist(), 'ent_score': ent_pre_prob[i].tolist(), 'ent_mask': ent_mask_np[i].tolist(),
            #              'query_entity': example_dict[cur_id].ques_entities_text, 'ctx_entity': example_dict[cur_id].ctx_entities_text,
            #              'ans_ent_mask': ans_cand_mask_np[i].tolist(), 'is_gold_ent': is_gold_ent_np[i].tolist(), 'answer': answer_dict[cur_id]}
            sent_pred_ = {'sp_score': predict_support_logit_np[i].tolist(), 'sp_mask': support_sent_mask_np[i].tolist(),
                          'sp_names': example_dict[cur_id].sent_names, 'sup_fact_id': orig_supp_fact_id, 'trim_sup_fact_id': prune_supp_fact_id}
            para_pred_ = {'para_score': predict_support_para_logit_np[i].tolist(), 'para_mask': support_para_mask_np[i].tolist(), 'para_names': example_dict[cur_id].para_names}
            ans_pred_ = {'ans_type': type_prob[i].tolist(), 'ent_score': ent_pre_logit_np[i].tolist(), 'ent_mask': ent_mask_np[i].tolist(),
                         'query_entity': example_dict[cur_id].ques_entities_text, 'ctx_entity': example_dict[cur_id].ctx_entities_text,
                         'ans_ent_mask': ans_cand_mask_np[i].tolist(), 'is_gold_ent': is_gold_ent_np[i].tolist(), 'answer': answer_dict[cur_id]}
            cls_emb_ = {'cls_emb': cls_emb_np[i].tolist()}
            res_pred = {**sent_pred_, **para_pred_, **ans_pred_, **cls_emb_}
            prediction_res_score_dict[cur_id] = res_pred
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
                total_sp_dict[thresh_i][cur_id].extend(post_process_thresh_i_sp_pred)
                # # +++++++++++++++++++++++++++
                # total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp_train.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = train_eval(tmp_file, train_gold_file, train_type)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold

    best_metrics, best_threshold = choose_best_threshold(answer_dict, prediction_file)
    json.dump(best_metrics, open(eval_file, 'w'))

    if output_score_file is not None:
        with open(output_score_file, 'w') as f:
            json.dump(prediction_res_score_dict, f)

    #####+++++++++++
    with open(train_gold_file) as f:
        gold = json.load(f)
    for row in gold:
        key = row['_id']
        print('suppo = {}'.format(row['supporting_facts']))
        score_case = prediction_res_score_dict[key]
        sp_names = score_case['sp_names']
        sup_fact_id = score_case['sup_fact_id']
        trim_sup_fact_id = score_case['trim_sup_fact_id']
        print('orig', [sp_names[_] for _ in sup_fact_id])
        print('trim', [sp_names[_] for _ in trim_sup_fact_id])
    #####+++++++++++

    print('Number of examples with cutted sentences = {}'.format(cut_sentence_count))
    return best_metrics, best_threshold
