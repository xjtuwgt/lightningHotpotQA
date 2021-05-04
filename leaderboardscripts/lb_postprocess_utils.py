import numpy as np
from numpy import ndarray
import json
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def distribution_feat_extraction(scores: ndarray, keep_num=False):
    ##     min, max, mean, median, 1/4, 3/4 score, std, num
    ##gap: min, max, mean, median, 1/4, 3/4 score, std
    min_value, max_value, mean_value, median_value, std_value = np.min(scores), np.max(scores), np.mean(scores), np.median(scores), np.std(scores)
    quartile_value_list = [2.5, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 75, 80, 90]
    quartile_feats = [np.percentile(scores, _) for _ in quartile_value_list]
    dist_feat = [min_value, max_value, mean_value, median_value, std_value] + quartile_feats
    dist_feat = [x.tolist() for x in dist_feat]
    if keep_num:
        num = float(scores.shape[0])
        dist_feat.append(num)
    return dist_feat

def distribution_feat(scores: ndarray):
    dist_feat = distribution_feat_extraction(scores=scores, keep_num=False)
    if scores.shape[0] > 1:
        sorted_score = np.sort(scores)
        reverse_score = sorted_score[::-1]
        gap_scores = np.array([reverse_score[i] - reverse_score[i+1] for i in range(scores.shape[0]-1)])
        gap_dist_feat = distribution_feat_extraction(scores=gap_scores, keep_num=False)
    else:
        gap_dist_feat = [0.0] * (len(dist_feat) - 1)
    dist_feat.extend(gap_dist_feat)
    return dist_feat

def row_x_feat_extraction(row):
    x_feats = row['cls_emb']
    # ++++++++++++++++++++++++++++++
    sent_scores = row['sp_score']
    sent_mask = row['sp_mask']
    sent_num = int(sum(sent_mask))
    sent_score_np = np.array(sent_scores[:sent_num])
    sent_feats = distribution_feat(scores=sent_score_np)
    x_feats += sent_feats
    return x_feats

def load_npz_data(npz_file_name):
    with np.load(npz_file_name) as data:
        x = data['x']
        y = data['y']
        y_n = data['y_n']
        y_np = data['y_np']
    print('Loading {} records from {}'.format(x.shape[0], npz_file_name))
    return x, y, y_n, y_np

def load_json_score_data(json_score_file_name):
    with open(json_score_file_name, 'r', encoding='utf-8') as reader:
        score_data = json.load(reader)
    print('Loading {} records from {}'.format(len(score_data), json_score_file_name))
    return score_data

###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.utils.data import Dataset
import numpy as np
class RangeDataset(Dataset):
    def __init__(self, json_file_name):
        self.feat_dict = load_json_score_data(json_score_file_name=json_file_name)
        self.key_list = list(self.feat_dict.keys())
    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        key = self.key_list[idx]
        case = self.feat_dict[key]
        x_feat = np.array(case['x_feat'])
        x_i = torch.from_numpy(x_feat).float()
        return key, x_i

    @staticmethod
    def collate_fn(data):
        x = torch.stack([_[1] for _ in data], dim=0)
        key = [_[0] for _ in data]
        sample = {'id': key, 'x_feat': x}
        return sample