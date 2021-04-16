from torch.utils.data import Dataset
from HotpotQAModel.hotpotqa_data_structure import Example
from torch.utils.data import DataLoader

class HotpotDataset(Dataset):
    def __init__(self, examples, max_para_num=4, max_sent_num=100, max_seq_num=512, sent_drop_ratio=0.25):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num
        self.sent_drop_ratio = sent_drop_ratio

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        case: Example = self.examples[idx]
