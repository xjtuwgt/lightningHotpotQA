import argparse
from envs import OUTPUT_FOLDER, DATASET_FOLDER
from numpy import ndarray
import numpy as np
from tqdm import tqdm
from hgntransformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import json
from os.path import join



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Adaptive threshold prediction')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--raw_dev_data", type=str, default='data_raw/hotpot_dev_distractor_v1.json')
    parser.add_argument("--raw_train_data", type=str, default='data_raw/hotpot_train_v1.1.json')
    parser.add_argument("--input_dir", type=str, default=DATASET_FOLDER, help='define output directory')
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    parser.add_argument("--pred_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    # Other parameters
    parser.add_argument("--train_type", type=str, default='hgn_low_sae', help='data type')
    parser.add_argument("--model_name_or_path", default='train.graph.roberta.bs2.as1.lr2e-05.lrslayer_decay.lrd0.9.gnngat1.4.datahgn_docred_low_saeRecAdam.cosine.seed103', type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--dev_score_name", type=str, default='dev_score.json')
    parser.add_argument("--train_score_name", type=str, default='train_score.json')

    parser.add_argument("--dev_feat_name", type=str, default='dev_np_data.npz')
    parser.add_argument("--dev_feat_class_name", type=str, default='dev_class_np_data.npz')
    parser.add_argument("--dev_feat_json_name", type=str, default='dev_json_data.json')
    parser.add_argument("--train_feat_name", type=str, default='train_np_data.npz')
    parser.add_argument("--train_feat_class_name", type=str, default='train_class_np_data.npz')
    parser.add_argument("--class_label_map_name", type=str, default='class_label_dict.json')
    parser.add_argument("--pred_threshold_json_name", type=str, default='pred_thresholds.json')

    parser.add_argument("--pickle_model_name", type=str, default='at_pred_model.pkl')
    parser.add_argument("--pickle_model_check_point_name", type=str, help='checkpoint name')

    return parser.parse_args(args)

def over_lap_ratio(ht_pair1, ref_ht_pair2):
    h, t = ht_pair1
    r_h, r_t = ref_ht_pair2
    if t < r_h or h > r_t: ## no overlap
        return 0.0, 1
    if r_h >= h and r_t <= t: ## subset: ref is a subset of given pair
        return 1.0, 2
    if r_h <= h and r_t >= t:
        return (t - h) / (r_t - r_h), 3 ## superset: ref is a superset of given pair
    if h >= r_h and h < r_t:
        return (r_t - h) / (r_t - r_h), 4
    if r_h >= h and r_h < t:
        return (t - r_h) / (r_t - r_h), 4


def distribution_feat_extraction(scores: ndarray, keep_num=False):
    ##     min, max, mean, median, 1/4, 3/4 score, std, num
    ##gap: min, max, mean, median, 1/4, 3/4 score, std
    min_value, max_value, mean_value, median_value, std_value = np.min(scores), np.max(scores), np.mean(scores), np.median(scores), np.std(scores)
    quartile_1, quartile_2 = np.percentile(scores, 25), np.percentile(scores, 75)
    quartile_3, quartile_4, quartile_5, quartile_6, quartile_7 = np.percentile(scores, 20), \
                                                                 np.percentile(scores, 40), \
                                                                 np.percentile(scores, 60), \
                                                                 np.percentile(scores, 80), \
                                                                 np.percentile(scores, 90)
    dist_feat = [min_value, max_value, mean_value, median_value, std_value, quartile_1, quartile_2, quartile_3, quartile_4, quartile_5, quartile_6, quartile_7]
    # dist_feat = [min_value, max_value, mean_value, median_value, std_value, quartile_1, quartile_2]
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
        gap_dist_feat = [0.0] * (len(dist_feat) - 1)
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

def feat_label_extraction(raw_data_name, score_data_name, train_type, train=False, train_filter=False):
    with open(raw_data_name, 'r', encoding='utf-8') as reader:
        raw_data = json.load(reader)
    print('Loading {} records from {}'.format(len(raw_data), raw_data_name))
    with open(score_data_name, 'r', encoding='utf-8') as reader:
        score_data = json.load(reader)
    print('Loading {} records from {}'.format(len(score_data), score_data_name))
    x_feats_list = []
    y_p_value_list = []
    y_n_value_list = []
    y_np_value_list = []

    x_feat_dict = {}
    for row_idx, row in tqdm(enumerate(raw_data)):
        qid = row['_id']
        if train:
            qid = qid + "_" + train_type
        if qid not in score_data:
            continue
        score_row = score_data[qid]
        x_feats = row_x_feat_extraction(row=score_row)
        flag, y_p, y_n = row_y_label_extraction(raw_row=row, score_row=score_row)
        if train and train_filter:
            if y_p < y_n:
                continue
        x_feats_list.append(x_feats)
        y_p_value_list.append(y_p)
        y_n_value_list.append(y_n)
        if y_p > y_n:
            y_np = (y_p + y_n)/2.0
        else:
            y_np = y_p
        y_np_value_list.append(y_np)
        x_feat_dict[qid] = x_feats
    assert len(x_feats_list) == len(y_p_value_list)
    print('Get {} features'.format(len(x_feats_list)))
    x_feats_np = np.array(x_feats_list)
    y_p_np = np.array(y_p_value_list)
    y_n_np = np.array(y_n_value_list)
    y_np_np = np.array(y_np_value_list)
    return x_feats_np, y_p_np, y_n_np, y_np_np, x_feat_dict

def save_numpy_array(x_feats: ndarray, y: ndarray, y_n: ndarray, y_np: ndarray, npz_file_name):
    np.savez(npz_file_name, x=x_feats, y=y, y_n=y_n, y_np=y_np)
    print('Saving {} records as x, and {} records as y into {}'.format(x_feats.shape, y.shape, npz_file_name))

def load_npz_data(npz_file_name):
    with np.load(npz_file_name) as data:
        x = data['x']
        y = data['y']
        y_n = data['y_n']
        y_np = data['y_np']
    print('Loading {} records from {}'.format(x.shape[0], npz_file_name))
    return x, y, y_n, y_np

def save_numpy_array_for_classification(x_feats: ndarray, y: ndarray, y_n: ndarray, y_np: ndarray, y_labels, npz_file_name):
    np.savez(npz_file_name, x=x_feats, y=y, y_n=y_n, y_np=y_np, y_label=y_labels)
    print('Saving {} records as x, and {} records as y into {}'.format(x_feats.shape, y_labels.shape, npz_file_name))

def load_npz_data_for_classification(npz_file_name):
    with np.load(npz_file_name) as data:
        x = data['x']
        y = data['y']
        y_n = data['y_n']
        y_np = data['y_np']
        y_labels = data['y_label']
    return x, y, y_n, y_np, y_labels

def threshold_map_to_label(y_p: ndarray, y_n: ndarray, threshold_category):
    over_lap_res = []
    for i in range(y_p.shape[0]):
        p_i = y_p[i]
        n_i = y_n[i]
        p_flag = True
        if p_i > n_i:
            ht_pair_i = (n_i, p_i)
        else:
            ht_pair_i = (p_i, n_i)
            p_flag = False
        over_lap_list = []
        for b_idx, bound in enumerate(threshold_category):
            over_lap_value, over_lap_type = over_lap_ratio(ht_pair_i, bound)
            over_lap_list.append((over_lap_value, over_lap_type))
        # print('p_i={}, n_i={}'.format(p_i, n_i))
        # print(over_lap_list)
        # print('*' * 100)
        over_lap_res.append((over_lap_list, p_flag))

    flag_list = []
    flag_label_freq = {}
    for i in range(y_p.shape[0]):
        # three_types = ''.join([str(int(x[1] > 1)) for x in over_lap_res[i][0]])
        three_types = ''.join([str(x[1]) for x in over_lap_res[i][0]])
        if over_lap_res[i][1]:
            flag_label = 'T_' + str(three_types)
        else:
            flag_label = 'F_' + str(three_types)
        if flag_label not in flag_label_freq:
            flag_label_freq[flag_label] = 1
        else:
            flag_label_freq[flag_label] = flag_label_freq[flag_label] + 1
        flag_list.append(flag_label)
    return flag_list, flag_label_freq

def adaptive_threshold_to_classification(train_npz_file_name, dev_npz_file_name, threshold_category, train_npz_class_file_name, dev_npz_class_file_name):
    train_x, train_y_p, train_y_n, train_y_np = load_npz_data(train_npz_file_name)
    dev_x, dev_y_p, dev_y_n, dev_y_np = load_npz_data(dev_npz_file_name)
    train_label_list, train_flag_label_freq_dict = threshold_map_to_label(y_p=train_y_p, y_n=train_y_n, threshold_category=threshold_category)
    for key, value in train_flag_label_freq_dict.items():
        print(key, value)
    print('Number of key words in train = {}'.format(len(train_flag_label_freq_dict)))
    print('*' * 75)
    dev_label_list, dev_flag_label_freq_dict = threshold_map_to_label(y_p=dev_y_p, y_n=dev_y_n, threshold_category=threshold_category)
    for key, value in dev_flag_label_freq_dict.items():
        print(key, value)
    print('Number of key words in dev = {}'.format(len(dev_flag_label_freq_dict)))
    flag_label_keys = sorted(list({**train_flag_label_freq_dict, **dev_flag_label_freq_dict}.keys()))
    for k_idx, key in enumerate(flag_label_keys):
        if key in train_flag_label_freq_dict and key in dev_flag_label_freq_dict:
            print('{}\t{}\t{}\t{}'.format(k_idx, key, train_flag_label_freq_dict[key] * 1.0 / train_y_p.shape[0],
                                      dev_flag_label_freq_dict[key] * 1.0 / dev_y_p.shape[0]))
        elif key in dev_flag_label_freq_dict:
            print('{}\t{}\t{}\t{}'.format(k_idx, key, 0, dev_flag_label_freq_dict[key] * 1.0 / dev_y_p.shape[0]))
        else:
            print('{}\t{}\t{}\t{}'.format(k_idx, key, train_flag_label_freq_dict[key] * 1.0 / train_y_p.shape[0], 0))
    label_key_to_idx_dict = dict([(key, k_idx) for k_idx, key in enumerate(flag_label_keys)])
    train_class_labels = np.array([label_key_to_idx_dict[_] for _ in train_label_list if _ in train_flag_label_freq_dict])
    dev_class_labels = np.array([label_key_to_idx_dict[_] for _ in dev_label_list if _ in dev_flag_label_freq_dict] )
    save_numpy_array_for_classification(x_feats=train_x, y=train_y_p, y_n=train_y_n, y_np=train_y_np, y_labels=train_class_labels, npz_file_name=train_npz_class_file_name)
    save_numpy_array_for_classification(x_feats=dev_x, y=dev_y_p, y_n=dev_y_n, y_np=train_y_np, y_labels=dev_class_labels, npz_file_name=dev_npz_class_file_name)
    return label_key_to_idx_dict


def get_optimizer(model, args):
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} ]
    print('Learning rate = {}'.format(args.learning_rate))
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer

def get_scheduler(optimizer, total_steps, args):
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=total_steps)
    return scheduler


def dev_data_collection(args):
    dev_raw_data_file_name = join(args.input_dir, args.raw_dev_data)
    dev_score_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_score_name)
    x_feats, y_value, y_n_np_value, y_np_value, x_feat_dict = feat_label_extraction(raw_data_name=dev_raw_data_file_name,
                                                                                    score_data_name=dev_score_file_name,
                                                                                    train_type=args.train_type, train=False)
    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    dev_json_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_json_name)
    save_numpy_array(x_feats=x_feats, y=y_value, y_n=y_n_np_value, y_np=y_np_value, npz_file_name=dev_npz_file_name)
    print('Saving dev data into {}'.format(dev_npz_file_name))
    json.dump(x_feat_dict, open(dev_json_file_name, 'w'))
    print('Saving dev data into {}'.format(dev_json_file_name))

def train_data_collection(args, train_filter):
    train_raw_data_file_name = join(args.input_dir, args.raw_train_data)
    train_score_file_name = join(args.pred_dir, args.model_name_or_path, args.train_type + '_' + args.train_score_name)
    if train_filter:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_name)
    else:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)
    x_feats, y_value, y_n_np_value, y_np_value, _ = feat_label_extraction(raw_data_name=train_raw_data_file_name, score_data_name=train_score_file_name,
                                             train_type=args.train_type, train=True, train_filter=train_filter)
    save_numpy_array(x_feats=x_feats, y=y_value, y_n=y_n_np_value, y_np=y_np_value, npz_file_name=train_npz_file_name)
    print('Saving train data into {}'.format(train_npz_file_name))


def train_dev_map_to_classification(args, train_filter, threshold_category):
    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    dev_class_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_class_name)

    if train_filter:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_name)
        train_class_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_class_name)
    else:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)
        train_class_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_class_name)

    class_label_dict = adaptive_threshold_to_classification(train_npz_file_name=train_npz_file_name, dev_npz_file_name=dev_npz_file_name,
                                         threshold_category=threshold_category, train_npz_class_file_name=train_class_npz_file_name,
                                         dev_npz_class_file_name=dev_class_npz_file_name)
    if train_filter:
        class_label_dict_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.class_label_map_name)
    else:
        class_label_dict_file_name = join(args.pred_dir, args.model_name_or_path, args.class_label_map_name)
    for key, value in class_label_dict.items():
        print(key, value)
    json.dump(class_label_dict, open(class_label_dict_file_name, 'w'))