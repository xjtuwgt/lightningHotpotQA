import json
import numpy as np
from tqdm import tqdm
import torch
import shutil
import os
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from csr_mhqa.utils import convert_to_tokens
import torch.nn.functional as F

def jd_unified_post_feature_collection_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file=None, output_score_file=None):
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    ##++++++
    prediction_res_score_dict = {}
    ##++++++

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
        cls_emb_np = cls_emb.data.cpu().numpy()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            orig_supp_fact_id = example_dict[cur_id].sup_fact_id
            prune_supp_fact_id = feature_dict[cur_id].sup_fact_ids
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            sent_pred_ = {'sp_score': predict_support_logit_np[i].tolist(), 'sp_mask': support_sent_mask_np[i].tolist(),
                          'sp_names': example_dict[cur_id].sent_names, 'sup_fact_id': orig_supp_fact_id,
                          'trim_sup_fact_id': prune_supp_fact_id}
            cls_emb_ = {'cls_emb': cls_emb_np[i].tolist()}
            res_score = {**sent_pred_, **cls_emb_}
            prediction_res_score_dict[cur_id] = res_score
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    if output_score_file is not None:
        with open(output_score_file, 'w') as f:
            json.dump(prediction_res_score_dict, f)
        print('Saving {} score records into {}'.format(len(prediction_res_score_dict), output_score_file))

    return best_metrics, best_threshold