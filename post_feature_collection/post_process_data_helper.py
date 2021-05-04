from leaderboardscripts.lb_postprocess_utils import load_json_score_data
import torch
import numpy as np
from torch.utils.data import Dataset

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
        if y_p_i > y_n_i:
            y_min = torch.FloatTensor([y_n_i])
            y_max = torch.FloatTensor([y_p_i])
            flag = torch.LongTensor([1])
        else:
            y_min = torch.FloatTensor([y_p_i])
            y_max = torch.FloatTensor([y_n_i])
            flag = torch.LongTensor([0])
        return x_i, y_min, y_max, flag, key

    @staticmethod
    def collate_fn(data):
        x = torch.stack([_[0] for _ in data], dim=0)
        y_min = torch.cat([_[1] for _ in data], dim=0)
        y_max = torch.cat([_[2] for _ in data], dim=0)
        flag = torch.cat([_[3] for _ in data], dim=0)
        key = [_[3] for _ in data]
        sample = {'x_feat': x, 'y_min': y_min, 'y_max': y_max, 'flag': flag, 'id': key}
        return sample