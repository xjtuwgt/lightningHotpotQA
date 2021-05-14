from leaderboardscripts.lb_postprocess_utils import load_json_score_data
import torch
import numpy as np
from post_feature_collection.post_process_feature_extractor import np_sigmoid
from torch.utils.data import Dataset
import random
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
        y_f1_score = y_label[0]
        x_i = torch.from_numpy(x_feat).float()
        y_p_i, y_n_i = y_label[1][1], y_label[1][0]

        # weight = torch.FloatTensor([np_sigmoid(y_p_i) - np_sigmoid(y_n_i)])
        weight = torch.FloatTensor([(np_sigmoid(y_p_i) - np_sigmoid(y_n_i)) * y_f1_score])
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
def trim_range(start_position, end_position, span_length, rand_ratio=0.25):
    seq_len = end_position - start_position + 1
    if seq_len <= span_length:
        return start_position, end_position
    else:
        span_list = []
        for i in range(seq_len - span_length):
            start_position_i = start_position + i
            end_position_i = start_position_i + span_length -1
            assert end_position_i <= end_position
            span_list.append((start_position_i, end_position_i))
        if random.random() < rand_ratio:
            rand_idx = random.randint(0, len(span_list) - 1)
            span_start, span_end = span_list[rand_idx]
        else:
            span_start, span_end = span_list[0]
        return span_start, span_end

class RangeSeqDataset(Dataset):
    def __init__(self, json_file_name, span_window_size):
        self.feat_dict = load_json_score_data(json_score_file_name=json_file_name)
        self.key_list = list(self.feat_dict.keys())
        self.span_window_size = span_window_size

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

        ##++++++
        l_idx, r_idx = trim_range(start_position=l_idx, end_position=r_idx, span_length=self.span_window_size)
        ##++++++

        y1 = torch.zeros(1, dtype=torch.long)
        y2 = torch.zeros(1, dtype=torch.long)

        y_label = case['y_label']
        y_f1_score = y_label[0]
        y_p_i, y_n_i = y_label[1][1], y_label[1][0]
        # weight = torch.FloatTensor([np_sigmoid(y_p_i) - np_sigmoid(y_n_i)])
        weight = torch.FloatTensor([(np_sigmoid(y_p_i) - np_sigmoid(y_n_i)) * y_f1_score])
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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def seq_drop(start_position, end_position, drop_ratio):
    seq_len = end_position - start_position + 1
    window_len = int(seq_len * (1-drop_ratio))
    if window_len == 0:
        return start_position, end_position

    return

class RangeSeqDropDataset(Dataset):
    def __init__(self, json_file_name, span_window_size, drop_ratio=0.25):
        self.feat_dict = load_json_score_data(json_score_file_name=json_file_name)
        self.key_list = list(self.feat_dict.keys())
        self.drop_ratio = drop_ratio
        self.span_window_size = span_window_size

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

        ##++++++
        l_idx, r_idx = trim_range(start_position=l_idx, end_position=r_idx, span_length=self.span_window_size)
        ##++++++

        y1 = torch.zeros(1, dtype=torch.long)
        y2 = torch.zeros(1, dtype=torch.long)

        y_label = case['y_label']
        y_f1_score = y_label[0]
        y_p_i, y_n_i = y_label[1][1], y_label[1][0]
        # weight = torch.FloatTensor([np_sigmoid(y_p_i) - np_sigmoid(y_n_i)])
        weight = torch.FloatTensor([(np_sigmoid(y_p_i) - np_sigmoid(y_n_i)) * y_f1_score])
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