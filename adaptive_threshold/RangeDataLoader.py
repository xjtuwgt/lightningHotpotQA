import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from adaptive_threshold.atutils import load_npz_data

class RangeDataset(Dataset):
    def __init__(self, npz_file_name):
        x, y_p, y_n, _ = load_npz_data(npz_file_name)
        self.x_feat = x
        self.y_p = y_p
        self.y_n = y_n

    def __len__(self):
        return self.x_feat.shape[0]

    def __getitem__(self, idx):
        x_i = torch.from_numpy(self.x_feat[idx])
        y_p_i, y_n_i = self.y_p[idx], self.y_n[idx]
        flag = True
        if y_p_i > y_n_i:
            y_min = torch.FloatTensor([y_n_i])
            y_max = torch.FloatTensor([y_p_i])
        else:
            y_min = torch.FloatTensor([y_p_i])
            y_max = torch.FloatTensor([y_n_i])
            flag = False
        return x_i, y_min, y_max, flag
