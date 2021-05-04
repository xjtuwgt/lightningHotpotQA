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
    return score_data



###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.utils.data import Dataset
class RangeDataset(Dataset):
    def __init__(self, npz_file_name):
        x, y_p, y_n, _ = load_npz_data(npz_file_name)
        self.x_feat = x
        self.y_p = y_p
        self.y_n = y_n
    def __len__(self):
        return self.x_feat.shape[0]

    def __getitem__(self, idx):
        x_i = torch.from_numpy(self.x_feat[idx]).float()
        y_p_i, y_n_i = self.y_p[idx], self.y_n[idx]
        if y_p_i > y_n_i:
            y_min = torch.FloatTensor([y_n_i])
            y_max = torch.FloatTensor([y_p_i])
            flag = torch.LongTensor([1])
        else:
            y_min = torch.FloatTensor([y_p_i])
            y_max = torch.FloatTensor([y_n_i])
            flag = torch.LongTensor([0])
        return x_i, y_min, y_max, flag

    @staticmethod
    def collate_fn(data):
        x = torch.stack([_[0] for _ in data], dim=0)
        y_min = torch.cat([_[1] for _ in data], dim=0)
        y_max = torch.cat([_[2] for _ in data], dim=0)
        flag = torch.cat([_[3] for _ in data], dim=0)
        sample = {'x_feat': x, 'y_min': y_min, 'y_max': y_max, 'flag': flag}
        return sample