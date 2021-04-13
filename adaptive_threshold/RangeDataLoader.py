import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from adaptive_threshold.atutils import load_npz_data

class RangeDataset(Dataset):

    def __init__(self, npz_file_name):
        x, y_p, y_n, train_y_np = load_npz_data(npz_file_name)
