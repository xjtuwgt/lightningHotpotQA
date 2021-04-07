import argparse
from envs import OUTPUT_FOLDER, DATASET_FOLDER
from numpy import ndarray
import numpy as np
from tqdm import tqdm

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
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default='train.graph.roberta.bs2.as1.lr2e-05.lrslayer_decay.lrd0.9.gnngat1.4.datahgn_docred_low_saeRecAdam.cosine.seed103', type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--dev_score_data", type=str, default='dev_score.json')
    parser.add_argument("--train_score_data", type=str, default='train_score.json')

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

def feat_label_extraction(data):
    for row_idx, row in tqdm(enumerate(data)):
        print(row_idx)