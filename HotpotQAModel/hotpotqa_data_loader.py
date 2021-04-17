from torch.utils.data import Dataset
from HotpotQAModel.hotpotqa_data_structure import Example
import numpy as np
from numpy import random
from torch.utils.data import DataLoader

class HotpotTrainDataset(Dataset):
    def __init__(self, examples, max_para_num=4, max_sent_num=100,
                 max_seq_num=512, sent_drop_ratio=0.25):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num
        self.sent_drop_ratio = sent_drop_ratio

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        case: Example = self.examples[idx]
        question_input_ids = case.question_input_ids
        ctx_input_ids = case.ctx_input_ids
        sent_num = case.sent_num
        para_num = case.para_num
        assert len(ctx_input_ids) == para_num


class HotpotDevDataset(Dataset):
    def __init__(self, examples, max_para_num=4, max_sent_num=100,
                 max_seq_num=512):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num

    def __getitem__(self, idx):
        case: Example = self.examples[idx]
        question_input_ids = case.question_input_ids
        ctx_input_ids = case.ctx_input_ids
        sent_num = case.sent_num
        para_num = case.para_num
        para_names = case.para_names
        sent_names = case.sent_names
        sup_para_ids = case.sup_para_id
        sup_sent_ids = case.sup_fact_id
        assert len(ctx_input_ids) == para_num

        doc_input_ids = question_input_ids
        doc_len = len(doc_input_ids)
        query_len = len(question_input_ids)


#######################################################################
def case_to_features(case: Example):
    question_input_ids = case.question_input_ids
    ctx_input_ids = case.ctx_input_ids
    sent_num = case.sent_num
    para_num = case.para_num
    para_names = case.para_names
    sent_names = case.sent_names
    assert len(ctx_input_ids) == para_num and sent_num == len(sent_names)
    sent_spans = []
    para_spans = []
    query_spans = []

    doc_input_ids = question_input_ids
    para_len_list = [len(question_input_ids)]
    sent_len_list = [len(question_input_ids)]
    query_len = len(question_input_ids)
    query_spans.append([0, query_len])
    for para_idx, para_name in enumerate(para_names):
        para_sent_ids = ctx_input_ids[para_idx]
        para_len_ = 0
        for sent_idx, sent_ids in enumerate(para_sent_ids):
            doc_input_ids += sent_ids
            sent_len_i = len(sent_ids)
            sent_len_list.append(sent_len_i)
            para_len_ = para_len_ + sent_len_i
        para_len_list.append(para_len_)
    assert sent_num == len(sent_len_list) - 1 and para_num == len(para_len_list) - 1
    sent_cum_sum_len_list = np.cumsum(sent_len_list).tolist()
    para_
    sent_spans = np.cumsum(sent_len_list).tolist()
    para_spans = np.cumsum(para_len_list).tolist()


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx
    return len(spans)

class HotpotTestDataset(Dataset):
    def __init__(self, examples, max_para_num=4, max_sent_num=100,
                 max_seq_num=512):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num

    def __getitem__(self, idx):
        case: Example = self.examples[idx]


