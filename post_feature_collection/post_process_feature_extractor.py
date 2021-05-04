from post_feature_collection.post_process_argument_parser import train_parser
from os.path import join
from leaderboardscripts.lb_postprocess_utils import load_json_score_data, row_x_feat_extraction
from tqdm import tqdm
import numpy as np

def feat_label_extraction(raw_data_name, score_data_name):
    raw_data = load_json_score_data(raw_data_name)
    print('Loading {} records from {}'.format(len(raw_data), raw_data_name))
    score_data = load_json_score_data(score_data_name)
    print('Loading {} records from {}'.format(len(score_data), score_data_name))

    for case in tqdm(raw_data):
        key = case['_id']
        score_case = score_data[key]
        # print(score_case)
        x_feat = row_x_feat_extraction(row=score_case)
        y_label = row_y_label_extraction(row=score_case)
        # print(y_label)


def train_feature_label_extraction(args):
    raw_train_file_name = join(args.input_dir, args.raw_train_data)
    train_score_file_name = join(args.output_dir, args.exp_name, args.train_score_name)
    train_feat_file_name = join(args.output_dir, args.exp_name, args.train_feat_json_name)
    feat_label_extraction(raw_data_name=raw_train_file_name, score_data_name=train_score_file_name)

    return

def dev_feature_label_extraction(args):
    raw_dev_file_name = join(args.input_dir, args.raw_dev_data)
    dev_score_file_name = join(args.output_dir, args.exp_name, args.dev_score_name)
    dev_feat_file_name = join(args.output_dir, args.exp_name, args.dev_feat_json_name)
    feat_label_extraction(raw_data_name=raw_dev_file_name, score_data_name=dev_score_file_name)
    return

def row_y_label_extraction(row):
    scores = row['sp_score']
    mask = row['sp_mask']
    supp_ids = row['trim_sup_fact_id']
    # get_best_f1_intervals(scores=scores, labels=supp_ids)
    num_candidate = int(sum(mask))
    labels = [0] * num_candidate
    for sup_id in supp_ids:
        labels[sup_id] = 1
    trim_scores = scores[:num_candidate]
    assert len(labels) == len(trim_scores)
    if len(supp_ids) == 0:
        sorted_scores = sorted(trim_scores, reverse=True)
        if len(sorted_scores) > 2:
            return (0, (sorted_scores[2], sorted_scores[1]))
        else:
            return (0, (None, None))
    # best_f1_interval(scores=scores, labels=labels)

def best_f1_interval(scores, labels):
    min_score, max_score = min(scores), max(scores)
    p_scores = [scores[idx] for idx, l in enumerate(labels) if l == 1]
    n_scores = [scores[idx] for idx, l in enumerate(labels) if l == 0]

    sorted_sl = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)


    thresholds = np.arange(min_score, max_score, 0.1).tolist()
    print(len(thresholds))
    # print(min_score, max_score)
    # print(sorted_sl)

def get_best_f1_intervals(scores, labels):
    best_f1_intervals = []
    for s, l in zip(scores, labels):
        l1 = [1 if i in l else 0 for i in range(len(s))]
        sorted_sl = sorted(zip(s, l1), key=lambda x: x[0])
        max_f1 = 0
        max_f1_scores = []
        for i in range(len(sorted_sl)):
            tp_left = sum([x[1] for x in sorted_sl[i:]])
            prec_left = tp_left / (len(sorted_sl) - i)
            recall_left = tp_left / (len(l) + 1e-6)
            f1_left = 2 * prec_left * recall_left / (prec_left + recall_left + 1e-6)
            if f1_left > max_f1 + 1e-6:
                max_f1 = f1_left
                max_f1_scores = [sorted_sl[i][0] - 1e-6]
            elif abs(f1_left - max_f1) < 1e-6:
                max_f1_scores.append(sorted_sl[i][0] - 1e-6)
            tp_right = sum([x[1] for x in sorted_sl[i+1:]])
            prec_right = tp_right / (len(sorted_sl) - i - 1 + 1e-6)
            recall_right = tp_right / (len(l) + 1e-6)
            f1_right = 2 * prec_right * recall_right / (prec_right + recall_right + 1e-6)
            if f1_right > max_f1 + 1e-6:
                max_f1 = f1_right
                max_f1_scores = [sorted_sl[i][0] + 1e-6]
            elif abs(f1_right - max_f1) < 1e-6:
                max_f1_scores.append(sorted_sl[i][0] + 1e-6)
        best_f1_intervals.append((max_f1, (min(max_f1_scores), max(max_f1_scores))))
    return best_f1_intervals

if __name__ == '__main__':
    args = train_parser()
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))
    print('*' * 75)
    # train_feature_label_extraction(args=args)
    dev_feature_label_extraction(args=args)