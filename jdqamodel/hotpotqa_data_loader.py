from torch.utils.data import Dataset
from jdqamodel.hotpotqa_data_structure import Example
from jdqamodel.hotpotqaUtils import example_sent_drop, case_to_features, trim_input_span
import torch
import numpy as np
from torch.utils.data import DataLoader
from os.path import join
from envs import DATASET_FOLDER
import gzip, pickle
from jdqamodel.hotpotqa_dump_features import get_cached_filename

IGNORE_INDEX = -100

class HotpotDataset(Dataset):
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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        query_mapping = torch.FloatTensor(trim_doc_segment_ids)
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
        sent_pad_num = self.max_sent_num - trim_sent_num
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
        supp_para_ids = [supp_para for supp_para in case.sup_para_id if supp_para < len(trim_para_spans)] ## supp para ids
        supp_sent_ids = [supp_sent for supp_sent in case.sup_fact_id if supp_sent < len(trim_sent_spans)] ## support fact ids
        is_support_sent = torch.zeros(self.max_sent_num, dtype=torch.float)
        for s_sent_id in supp_sent_ids:
            is_support_sent[s_sent_id] = 1
        is_gold_para = torch.zeros(self.max_para_num, dtype=torch.float)
        for s_para_id in supp_para_ids:
            is_gold_para[s_para_id] = 1
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        q_type = ans_type_label
        if q_type[0] < 2:
            y1, y2 = [IGNORE_INDEX], [IGNORE_INDEX]
        else:
            if len(trim_ans_spans) == 0:
                y1, y2 = [IGNORE_INDEX], [IGNORE_INDEX]
            elif len(trim_ans_spans) == 1:
                y1, y2 = [trim_ans_spans[0][0]], [trim_ans_spans[0][1]]
            else:
                rand_idx = np.random.randint(len(trim_ans_spans))
                y1, y2 = [trim_ans_spans[rand_idx][0]], [trim_ans_spans[rand_idx][1]]
        q_type = torch.LongTensor(q_type)
        y1 = torch.LongTensor(y1)
        y2 = torch.LongTensor(y2)

        res = {'ids': id,
               'y1': y1,
               'y2': y2,
               'q_type': q_type,
               'is_support': is_support_sent,
               'is_gold_para': is_gold_para,
               'context_idxs': trim_doc_input_ids,
               'context_mask': trim_doc_input_mask,
               'segment_idxs': trim_doc_segment_ids,
               'context_lens': trim_doc_input_length,
               'query_start': query_start_position,
               'query_end': query_end_position,
               'query_mapping': query_mapping,
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
        assert len(data[0]) == 21
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
            elif key in {'context_lens', 'para_num', 'sent_num'}:
                batch_data[key] = torch.LongTensor([_[key] for _ in data])
            elif key in {'q_type', 'y1', 'y2'}:
                batch_data[key] = torch.cat([_[key] for _ in data], dim=0)
            else:
                batch_data[key] = torch.stack([_[key] for _ in data], dim=0)
        trim_keys = ['context_idxs', 'context_mask', 'segment_idxs']
        for key in trim_keys:
            batch_data[key] = batch_data[key][:, :max_c_len]
        return batch_data


class HotpotTestDataset(Dataset):
    def __init__(self, examples, sep_token_id, max_para_num=4, max_sent_num=100, max_seq_num=512):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num
        self.sep_token_id = sep_token_id

    def __len__(self):
        return len(self.examples)

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
        query_mapping = torch.FloatTensor(trim_doc_segment_ids)
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
        sent_pad_num = self.max_sent_num - trim_sent_num
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
        res = {'ids': id,
            'context_idxs': trim_doc_input_ids,
            'context_mask': trim_doc_input_mask,
            'segment_idxs': trim_doc_segment_ids,
            'context_lens': trim_doc_input_length,
            'query_start': query_start_position,
            'query_end': query_end_position,
            'query_mapping': query_mapping,
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
        assert len(data[0]) == 16
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
            elif key in {'context_lens', 'para_num', 'sent_num'}:
                batch_data[key] = torch.LongTensor([_[key] for _ in data])
            else:
                batch_data[key] = torch.stack([_[key] for _ in data], dim=0)
        trim_keys = ['context_idxs', 'context_mask', 'segment_idxs']
        for key in trim_keys:
            batch_data[key] = batch_data[key][:, :max_c_len]
        return batch_data

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DataHelper:
    def __init__(self, sep_token_id, gz=True, config=None):
        self.Dataset = HotpotDataset
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')

        self.__train_examples__ = None
        self.__dev_examples__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None

        self.config = config
        self.sep_token_id = sep_token_id
        self.train_f_type = self.config.daug_type

    def get_example_file(self, tag, f_type=None):
        cached_filename = get_cached_filename('{}_hotpotqa_tokenized_examples'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    @property
    def train_example_file(self):
        return self.get_example_file('train', self.train_f_type)

    @property
    def dev_example_file(self):
        return self.get_example_file('dev_distractor', self.config.devf_type)

    def get_pickle_file(self, file_name):
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        if getattr(self, name) is None:
            with self.get_pickle_file(file) as fin:
                print('loading', file)
                setattr(self, name, pickle.load(fin))
        return getattr(self, name)

    # Examples
    @property
    def train_examples(self):
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)

    # Example dict
    @property
    def train_example_dict(self):
        if self.__train_example_dict__ is None:
            self.__train_example_dict__ = {e.qas_id: e for e in self.train_examples}
        return self.__train_example_dict__

    @property
    def dev_example_dict(self):
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    # Load
    def load_dev(self):
        return self.dev_examples, self.dev_example_dict

    def load_train(self):
        return self.train_examples, self.train_example_dict

    @property
    def hotpot_train_dataloader(self) -> DataLoader:
        train_data_examples, _ = self.load_train()
        train_data = self.Dataset(examples=train_data_examples,
                                  max_para_num=self.config.max_para_num,
                                  max_sent_num=self.config.max_sent_num,
                                  max_seq_num=self.config.max_seq_length,
                                  sep_token_id=self.sep_token_id,
                                  sent_drop_ratio=self.config.sent_drop_ratio)
        ####++++++++++++
        dataloader = DataLoader(dataset=train_data, batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=HotpotDataset.collate_fn)
        return dataloader

    @property
    def hotpot_val_dataloader(self) -> DataLoader:
        dev_data_examples, _ = self.load_dev()
        dev_data = self.Dataset(examples=dev_data_examples,
                                  max_para_num=self.config.max_para_num,
                                  max_sent_num=self.config.max_sent_num,
                                  max_seq_num=self.config.max_seq_length,
                                  sep_token_id=self.sep_token_id,
                                  sent_drop_ratio=-1.0)
        ####++++++++++++
        dataloader = DataLoader(
            dataset=dev_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=HotpotDataset.collate_fn
        )
        return dataloader