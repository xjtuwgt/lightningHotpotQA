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
from leaderboardscripts.lb_postprocess_utils import row_x_feat_extraction
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
        cls_emb_np = cls_emb.data.cpu().numpy()
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        for i in range(predict_support_np.shape[0]):
            cur_id = batch['ids'][i]
            sp_dict[cur_id] = []
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_pred_ = {'sp_score': predict_support_logit_np[i].tolist(), 'sp_mask': support_sent_mask_np[i].tolist(),
                          'sp_names': example_dict[cur_id].sent_names}
            cls_emb_ = {'cls_emb': cls_emb_np[i].tolist()}
            res_score = {**sent_pred_, **cls_emb_}
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

def jd_adaptive_threshold_post_process(args, full_file, prediction_answer_file, score_dict_file, threshold_pred_dict_file, eval_file=None, dev_gold_file=None):
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
    print('Loading {} records from {}'.format(len(threshold_pred_dict), threshold_pred_dict))

    pred_answer = pred_data['answer']
    pred_type = pred_data['type']

    for case in tqdm(full_data):
        key = case['_id']
        score_case = score_dict[key]
        threshold_case = threshold_pred_dict[key]

    return


def jd_unified_eval_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file=None):
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