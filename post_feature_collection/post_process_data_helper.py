from leaderboardscripts.lb_postprocess_utils import load_json_score_data
import torch
import numpy as np
from post_feature_collection.post_process_feature_extractor import np_sigmoid
from torch.utils.data import Dataset
IGNORE_INDEX = -100

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
        y_label = case['y_label']
        x_i = torch.from_numpy(x_feat).float()
        y_p_i, y_n_i = y_label[1][1], y_label[1][0]
        weight = torch.FloatTensor([1 - np_sigmoid(y_p_i) + np_sigmoid(y_n_i)])
        y_min = torch.FloatTensor([y_n_i])
        y_max = torch.FloatTensor([y_p_i])
        if y_label[0] == 1.0:
            flag = torch.LongTensor([1])
        else:
            flag = torch.LongTensor([0])
        return x_i, y_min, y_max, weight, flag, key

    @staticmethod
    def collate_fn(data):
        x = torch.stack([_[0] for _ in data], dim=0)
        y_min = torch.cat([_[1] for _ in data], dim=0)
        y_max = torch.cat([_[2] for _ in data], dim=0)
        weight = torch.cat([_[3] for _ in data], dim=0)
        flag = torch.cat([_[4] for _ in data], dim=0)
        key = [_[5] for _ in data]
        sample = {'x_feat': x, 'y_min': y_min, 'weight': weight, 'y_max': y_max, 'flag': flag, 'id': key}
        return sample

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class RangeSeqDataset(Dataset):
    def __init__(self, json_file_name):
        self.feat_dict = load_json_score_data(json_score_file_name=json_file_name)
        self.key_list = list(self.feat_dict.keys())

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        key = self.key_list[idx]
        case = self.feat_dict[key]
        x_feat = np.array(case['x_feat'])
        y_seq_label = case['y_seq_label']
        _, p_flag, seq_label = y_seq_label
        x_i = torch.from_numpy(x_feat).float()
        l_idx = seq_label.find('2') - 2
        r_idx = seq_label.rfind('2') - 2

        y1 = torch.zeros(1, dtype=torch.long)
        y2 = torch.zeros(1, dtype=torch.long)

        y_label = case['y_label']
        y_p_i, y_n_i = y_label[1][1], y_label[1][0]
        # weight = torch.FloatTensor([y_p_i - y_n_i])
        weight = torch.FloatTensor([1 - np_sigmoid(y_p_i) + np_sigmoid(y_n_i)])
        y_min = torch.FloatTensor([y_n_i])
        y_max = torch.FloatTensor([y_p_i])


        if l_idx < 0:
            y1[0] = IGNORE_INDEX
            y2[0] = IGNORE_INDEX
        else:
            y1[0] = l_idx
            y2[0] = r_idx
        if p_flag:
            flag = torch.LongTensor([1])
        else:
            flag = torch.LongTensor([0])
        return x_i, y1, y2, y_min, y_max, weight, flag, key
    @staticmethod
    def collate_fn(data):
        x = torch.stack([_[0] for _ in data], dim=0)
        y_1 = torch.cat([_[1] for _ in data], dim=0)
        y_2 = torch.cat([_[2] for _ in data], dim=0)
        y_min = torch.cat([_[3] for _ in data], dim=0)
        y_max = torch.cat([_[4] for _ in data], dim=0)
        weight = torch.cat([_[5] for _ in data], dim=0)
        flag = torch.cat([_[6] for _ in data], dim=0)
        key = [_[7] for _ in data]
        sample = {'x_feat': x, 'y_1': y_1, 'y_2': y_2, 'y_min': y_min, 'y_max': y_max, 'weight': weight, 'flag': flag, 'id': key}
        return sample