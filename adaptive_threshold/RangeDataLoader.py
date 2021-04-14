import torch
from torch.utils.data import Dataset
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