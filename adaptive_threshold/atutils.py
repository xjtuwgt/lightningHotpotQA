import argparse
from envs import OUTPUT_FOLDER, DATASET_FOLDER
from numpy import ndarray
import numpy as np
from tqdm import tqdm
import json

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Adaptive threshold prediction')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--raw_dev_data", type=str, default='data_raw/hotpot_dev_distractor_v1.json')
    parser.add_argument("--raw_train_data", type=str, default='data_raw/hotpot_dev_distractor_v1.json')
    parser.add_argument("--input_dir", type=str, default=DATASET_FOLDER, help='define output directory')
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    parser.add_argument("--pred_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    # Other parameters
    parser.add_argument("--train_type", type=str, default='hgn_low_sae', help='data type')
    parser.add_argument("--model_name_or_path", default='train.graph.roberta.bs2.as1.lr2e-05.lrslayer_decay.lrd0.9.gnngat1.4.datahgn_docred_low_saeRecAdam.cosine.seed103', type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--dev_score_name", type=str, default='dev_score.json')
    parser.add_argument("--train_score_name", type=str, default='train_score.json')

    return parser.parse_args(args)


def distribution_feat_extraction(scores: ndarray, keep_num=False):
    ##     min, max, mean, median, 1/4, 3/4 score, std, num
    ##gap: min, max, mean, median, 1/4, 3/4 score, std
    min_value, max_value, mean_value, median_value, std_value = np.min(scores), np.max(scores), np.mean(scores), np.median(scores), np.std(scores)
    quartile_1, quartile_2 = np.percentile(scores, 25), np.percentile(scores, 75)
    dist_feat = [min_value, max_value, mean_value, median_value, std_value, quartile_1, quartile_2]
    dist_feat = [x.tolist() for x in dist_feat]
    if keep_num:
        num = float(scores.shape[0])
        dist_feat.append(num)
    return dist_feat

def distribution_feat(scores: ndarray):
    dist_feat = distribution_feat_extraction(scores=scores, keep_num=True)
    if scores.shape[0] > 1:
        sorted_score = np.sort(scores)
        reverse_score = sorted_score[::-1]
        gap_scores = np.array([reverse_score[i] - reverse_score[i+1] for i in range(scores.shape[0]-1)])
        gap_dist_feat = distribution_feat_extraction(scores=gap_scores, keep_num=False)
    else:
        gap_dist_feat = [0.0] * 7
    dist_feat.extend(gap_dist_feat)
    return dist_feat

def row_y_label_extraction(raw_row, score_row):
    def positive_neg_score(scores, mask, names, gold_names):
        assert len(scores) == len(mask)
        mask_sum_num = int(sum(mask))
        prune_names = names[:mask_sum_num]
        gold_name_set = set(gold_names)
        if (gold_name_set.issubset(set(prune_names))):
            flag = True
        else:
            flag = False
        positive_scores = []
        negative_scores = []
        for idx in range(mask_sum_num):
            name_i = prune_names[idx]
            if name_i in gold_name_set:
                positive_scores.append(scores[idx])
            else:
                negative_scores.append(scores[idx])

        if len(positive_scores) > 0:
            min_positive = min(positive_scores)
        else:
            min_positive = 0.0
        if len(negative_scores) == 0:
            max_negative = 1.0
        else:
            max_negative = max(negative_scores)

        return flag, min_positive, max_negative

    sp_golds = raw_row['supporting_facts']
    sp_golds = [(x[0], x[1]) for x in sp_golds]
    sp_scores = score_row['sp_score']
    sp_mask = score_row['sp_mask']
    sp_names = score_row['sp_names']
    sp_names = [(x[0], x[1]) for x in sp_names]
    flag, min_positive, max_negative = positive_neg_score(scores=sp_scores, mask=sp_mask, names=sp_names, gold_names=sp_golds)
    return flag, min_positive, max_negative

def row_x_feat_extraction(row):
    x_feats = row['cls_emb']
    #++++++++++++++
    para_scores = row['para_score']
    para_mask = row['para_mask']
    para_num = int(sum(para_mask))
    para_score_np = np.array(para_scores[:para_num])
    para_feats = distribution_feat(scores=para_score_np)
    x_feats += para_feats
    # ++++++++++++++
    sent_scores = row['sp_score']
    sent_mask = row['sp_mask']
    sent_num = int(sum(sent_mask))
    sent_score_np = np.array(sent_scores[:sent_num])
    sent_feats = distribution_feat(scores=sent_score_np)
    x_feats += sent_feats
    # ++++++++++++++
    ent_scores = row['ent_score']
    ent_mask = row['ent_mask']
    ent_num = int(sum(ent_mask))
    ent_score_np = np.array(ent_scores[:ent_num])
    ent_feats = distribution_feat(scores=ent_score_np)
    x_feats += ent_feats
    return x_feats

def feat_label_extraction(raw_data_name, score_data_name, train_type, train=False):
    with open(raw_data_name, 'r', encoding='utf-8') as reader:
        row_data = json.load(reader)
    with open(score_data_name, 'r', encoding='utf-8') as reader:
        score_data = json.load(reader)
    x_feats_list = []
    y_value_list = []
    for row_idx, row in tqdm(enumerate(row_data)):
        qid = row['_id']
        score_row = score_data[qid]
        x_feats = row_x_feat_extraction(row=score_row)
        x_feats_list.append(x_feats)
        flag, y_p, y_n = row_y_label_extraction(raw_row=row, score_row=score_row)
        y_value_list.append(y_p)
    assert len(x_feats_list) == len(y_value_list)
    y_value_np = np.array(y_value_list)
    s_y = np.sort(y_value_np)
    for i in range(len(x_feats_list)):
        print(i, s_y[i])