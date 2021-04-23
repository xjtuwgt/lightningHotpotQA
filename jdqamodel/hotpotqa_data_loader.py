from torch.utils.data import Dataset
from jdqamodel.hotpotqa_data_structure import Example
from jdqamodel.hotpotqaUtils import example_sent_drop, case_to_features, trim_input_span
import torch
import numpy as np
from torch.utils.data import DataLoader

class HotpotTrainDataset(Dataset):
    def __init__(self, examples, sep_token_id, max_para_num=4, max_sent_num=100,
                 max_seq_num=512, sent_drop_ratio=0.25):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num
        self.sep_token_id = sep_token_id
        self.sent_drop_ratio = sent_drop_ratio

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        case: Example = self.examples[idx]
        if self.sent_drop_ratio > 0:
            case = example_sent_drop(case=case, drop_ratio=self.sent_drop_ratio)
        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = \
            case_to_features(case=case, train_dev=True)
        trim_doc_input_ids, trim_query_spans, trim_para_spans, trim_sent_spans, trim_ans_spans = trim_input_span(
            doc_input_ids, query_spans, para_spans, sent_spans, limit=self.max_seq_length, sep_token_id=self.sep_token_id, ans_spans=ans_spans)
        supp_para_ids = case.sup_para_id ## support paraids
        supp_sent_ids = case.sup_fact_id


class HotpotDevDataset(Dataset):
    def __init__(self, examples, sep_token_id, max_para_num=4, max_sent_num=100,
                 max_seq_num=512):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num
        self.sep_token_id = sep_token_id

    def __getitem__(self, idx):
        case: Example = self.examples[idx]
        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = \
            case_to_features(case=case, train_dev=True)
        trim_doc_input_ids, trim_query_spans, trim_para_spans, trim_sent_spans, trim_ans_spans = trim_input_span(
            doc_input_ids, query_spans, para_spans, sent_spans,
            limit=self.max_seq_length, sep_token_id=self.sep_token_id, ans_spans=ans_spans)

class HotpotTestDataset(Dataset):
    def __init__(self, examples, sep_token_id, max_para_num=4, max_sent_num=100, max_seq_num=512):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num
        self.sep_token_id = sep_token_id

    def __getitem__(self, idx):
        case: Example = self.examples[idx]
        doc_input_ids, query_spans, para_spans, sent_spans = \
            case_to_features(case=case, train_dev=False)
        trim_doc_input_ids, trim_query_spans, trim_para_spans, trim_sent_spans = trim_input_span(
            doc_input_ids, query_spans, para_spans, sent_spans,
            limit=self.max_seq_length, sep_token_id=self.sep_token_id)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        trim_doc_input_length = len(trim_doc_input_ids)
        trim_doc_input_mask = [1] * trim_doc_input_length
        trim_doc_segment_ids = [0] * trim_query_spans[0][1] + [1] * (trim_doc_input_length - trim_query_spans[0][1])
        doc_pad_length = self.max_seq_length - trim_doc_input_length
        trim_doc_input_ids += [0] * doc_pad_length
        trim_doc_input_mask += [0] * doc_pad_length
        trim_doc_segment_ids += [0] * doc_pad_length
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        assert len(trim_doc_input_ids) == self.max_seq_length
        assert len(trim_doc_input_mask) == self.max_seq_length
        assert len(trim_doc_segment_ids) == self.max_seq_length
        trim_doc_input_ids = torch.LongTensor(trim_doc_input_ids)
        trim_doc_input_mask = torch.LongTensor(trim_doc_input_mask)
        trim_doc_segment_ids = torch.LongTensor(trim_doc_segment_ids)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_start_position, query_end_position = [trim_query_spans[0][0]], [trim_query_spans[0][1]]
        query_start_position = torch.LongTensor(query_start_position)
        query_end_position = torch.LongTensor(query_end_position)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        trim_para_num = len(trim_para_spans)
        trim_para_mask = [1] * trim_para_num
        para_pad_num = self.max_para_num - trim_para_num
        trim_para_mask += [0] * para_pad_num
        trim_para_start_position = [_[0] for _ in trim_para_spans]
        trim_para_end_position = [_[1] for _ in trim_para_spans]
        trim_para_start_position += [0] * para_pad_num
        trim_para_end_position += [0] * para_pad_num
        assert len(trim_para_start_position) == self.max_para_num
        assert len(trim_para_end_position) == self.max_para_num
        assert len(trim_para_mask) == self.max_para_num
        trim_para_start_position = torch.LongTensor(trim_para_start_position)
        trim_para_end_position = torch.LongTensor(trim_para_end_position)
        trim_para_mask = torch.LongTensor(trim_para_mask)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if len(trim_sent_spans) > self.max_sent_num:
            trim_sent_spans = trim_sent_spans[:self.max_sent_num]
        trim_sent_num = len(trim_sent_spans)
        assert trim_sent_num <= self.max_sent_num
        trim_sent_mask = [1] * trim_sent_num
        sent_pad_num = self.max_sent_num
        trim_sent_mask += [0] * sent_pad_num
        trim_sent_start_position = [_[0] for _ in trim_sent_spans]
        trim_sent_end_position = [_[1] for _ in trim_sent_spans]
        trim_sent_start_position += [0] * sent_pad_num
        trim_sent_end_position += [0] * sent_pad_num
        assert len(trim_sent_start_position) == self.max_sent_num
        assert len(trim_sent_end_position) == self.max_sent_num
        assert len(trim_sent_mask) == self.max_sent_num
        trim_sent_start_position = torch.LongTensor(trim_sent_start_position)
        trim_sent_end_position = torch.LongTensor(trim_sent_end_position)
        trim_sent_mask = torch.LongTensor(trim_sent_mask)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        id = case.qas_id
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        res = {
            'ids': id,
            'context_idxs': trim_doc_input_ids,
            'context_mask': trim_doc_input_mask,
            'segment_idxs': trim_doc_segment_ids,
            'context_lens': trim_doc_input_length,
            'query_start': query_start_position,
            'query_end': query_end_position,
            'para_start': trim_para_start_position,
            'para_end': trim_para_end_position,
            'para_mask': trim_para_mask,
            'para_num': trim_para_num,
            'sent_start': trim_sent_start_position,
            'sent_end': trim_sent_end_position,
            'sent_mask': trim_sent_mask,
            'sent_num': trim_sent_num}
        return res

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 14
        context_lens_np = np.array([_['context_lens'] for _ in data])
        max_c_len = context_lens_np.max()
        sorted_idxs = np.argsort(context_lens_np)[::-1]
        assert len(data) == len(sorted_idxs)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        data = [data[_] for _ in sorted_idxs]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        data_keys = data[0].keys()
        batch_data = {}
        for key in data_keys:
            if key in {'ids'}:
                batch_data[key] = [_[key] for _ in data]
            elif key in {'context_lens'}:
                batch_data[key] = torch.LongTensor([_[key] for _ in data])
            else:
                batch_data[key] = torch.stack([_[key] for _ in data], dim=0)
        trim_keys = ['context_idxs', 'context_mask', 'segment_idxs']
        for key in trim_keys:
            batch_data[key] = batch_data[key][:, :max_c_len]
        return batch_data

