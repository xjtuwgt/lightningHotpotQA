import gzip
import pickle
import torch
import numpy as np

from os.path import join
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from envs import DATASET_FOLDER

IGNORE_INDEX = -100

def get_cached_filename(f_type, config):
    f_type_set = {'examples', 'features', 'graphs', 'hgn_examples',
                  'hgn_features', 'hgn_graphs', 'hgn_reverse_examples', 'hgn_reverse_features',
                  'hgn_reverse_graphs', 'long_examples',
                  'long_features', 'long_graphs', 'long_reverse_examples', 'long_reverse_features',
                  'long_reverse_graphs'}
    assert f_type in f_type_set
    return f"cached_{f_type}_{config.model_type}_{config.max_seq_length}_{config.max_query_length}.pkl.gz"

class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 sup_para_id,
                 ques_entities_text,
                 ctx_entities_text,
                 para_start_end_position,
                 sent_start_end_position,
                 ques_entity_start_end_position,
                 ctx_entity_start_end_position,
                 question_text,
                 question_word_to_char_idx,
                 ctx_text,
                 ctx_word_to_char_idx,
                 edges=None,
                 orig_answer_text=None,
                 answer_in_ques_entity_ids=None,
                 answer_in_ctx_entity_ids=None,
                 answer_candidates_in_ctx_entity_ids=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.sup_para_id = sup_para_id
        self.ques_entities_text = ques_entities_text
        self.ctx_entities_text = ctx_entities_text
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.ques_entity_start_end_position = ques_entity_start_end_position
        self.ctx_entity_start_end_position = ctx_entity_start_end_position
        self.question_word_to_char_idx = question_word_to_char_idx
        self.ctx_text = ctx_text
        self.ctx_word_to_char_idx = ctx_word_to_char_idx
        self.edges = edges
        self.orig_answer_text = orig_answer_text
        self.answer_in_ques_entity_ids = answer_in_ques_entity_ids
        self.answer_in_ctx_entity_ids = answer_in_ctx_entity_ids
        self.answer_candidates_in_ctx_entity_ids= answer_candidates_in_ctx_entity_ids
        self.start_position = start_position
        self.end_position = end_position

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 para_spans,
                 sent_spans,
                 entity_spans,
                 q_entity_cnt,
                 sup_fact_ids,
                 sup_para_ids,
                 ans_type,
                 token_to_orig_map,
                 edges=None,
                 orig_answer_text=None,
                 answer_in_entity_ids=None,
                 answer_candidates_ids=None,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.para_spans = para_spans
        self.sent_spans = sent_spans
        self.entity_spans = entity_spans
        self.q_entity_cnt = q_entity_cnt
        self.sup_fact_ids = sup_fact_ids
        self.sup_para_ids = sup_para_ids
        self.ans_type = ans_type

        self.edges = edges
        self.token_to_orig_map = token_to_orig_map
        self.orig_answer_text = orig_answer_text
        self.answer_in_entity_ids = answer_in_entity_ids
        self.answer_candidates_ids = answer_candidates_ids

        self.start_position = start_position
        self.end_position = end_position

class DataHelper:
    def __init__(self, gz=True, config=None, f_type=None):
        self.Dataset = HotpotDataset
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')

        self.__train_features__ = None
        self.__dev_features__ = None

        self.__train_examples__ = None
        self.__dev_examples__ = None

        self.__train_graphs__ = None
        self.__dev_graphs__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None

        self.config = config

        self.f_type = f_type

    def get_feature_file(self, tag):
        if self.f_type is None:
            cached_filename = get_cached_filename('features', self.config)
        else:
            cached_filename = get_cached_filename('{}_features'.format(self.f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_example_file(self, tag):
        if self.f_type is None:
            cached_filename = get_cached_filename('examples', self.config)
        else:
            cached_filename = get_cached_filename('{}_examples'.format(self.f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_graph_file(self, tag):
        if self.f_type is None:
            cached_filename = get_cached_filename('graphs', self.config)
        else:
            cached_filename = get_cached_filename('{}_graphs'.format(self.f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    @property
    def train_feature_file(self):
        return self.get_feature_file('train')

    @property
    def dev_feature_file(self):
        return self.get_feature_file('dev_distractor')

    @property
    def train_example_file(self):
        return self.get_example_file('train')

    @property
    def dev_example_file(self):
        return self.get_example_file('dev_distractor')

    @property
    def train_graph_file(self):
        return self.get_graph_file('train')

    @property
    def dev_graph_file(self):
        return self.get_graph_file('dev_distractor')

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

    # Features
    @property
    def train_features(self):
        return self.__get_or_load__('__train_features__', self.train_feature_file)

    @property
    def dev_features(self):
        return self.__get_or_load__('__dev_features__', self.dev_feature_file)

    # Examples
    @property
    def train_examples(self):
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)

    # Graphs
    @property
    def train_graphs(self):
        return self.__get_or_load__('__train_graphs__', self.train_graph_file)

    @property
    def dev_graphs(self):
        return self.__get_or_load__('__dev_graphs__', self.dev_graph_file)

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

    # Feature dict
    @property
    def train_feature_dict(self):
        return {e.qas_id: e for e in self.train_features}

    @property
    def dev_feature_dict(self):
        return {e.qas_id: e for e in self.dev_features}

    # Load
    def load_dev(self):
        return self.dev_features, self.dev_example_dict, self.dev_graphs

    def load_train(self):
        return self.train_features, self.train_example_dict, self.train_graphs

    @property
    def dev_loader(self):
        return self.Dataset(*self.load_dev(),
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 ans_ent_limit=self.config.max_ans_ent_num,
                                 mask_edge_types=self.config.mask_edge_types)

    @property
    def train_loader(self):
        return self.Dataset(*self.load_train(),
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 ans_ent_limit=self.config.max_ans_ent_num,
                                 mask_edge_types=self.config.mask_edge_types)

    @property
    def hotpot_train_dataloader(self) -> DataLoader:
        train_data = self.Dataset(*self.load_train(),
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 ans_ent_limit=self.config.max_ans_ent_num,
                                 mask_edge_types=self.config.mask_edge_types)
        dataloader = DataLoader(dataset=train_data, batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=HotpotDataset.collate_fn)
        return dataloader

    @property
    def hotpot_val_dataloader(self) -> DataLoader:
        dev_data = self.Dataset(*self.load_dev(),
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 ans_ent_limit=self.config.max_ans_ent_num,
                                 mask_edge_types=self.config.mask_edge_types)
        dataloader = DataLoader(
            dataset=dev_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=HotpotDataset.collate_fn
        )
        return dataloader


class HotpotDataset(Dataset):
    def __init__(self, features, example_dict, graph_dict, para_limit, sent_limit, ent_limit, ans_ent_limit,
                 mask_edge_types):
        self.features = features
        self.example_dict = example_dict
        self.graph_dict = graph_dict
        print(len(self.features), type(self.features), len(self.example_dict), type(self.example_dict), len(self.graph_dict), type(self.graph_dict))
        self.para_limit = para_limit
        self.sent_limit = sent_limit
        self.ent_limit = ent_limit
        self.ans_ent_limit = ans_ent_limit
        self.graph_nodes_num = 1 + para_limit + sent_limit + ent_limit
        self.mask_edge_types = mask_edge_types
        self.max_seq_length = 512

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # BERT input
        context_idxs = torch.zeros(1, self.max_seq_length, dtype=torch.long)
        context_mask = torch.zeros(1, self.max_seq_length, dtype=torch.long)
        segment_idxs = torch.zeros(1, self.max_seq_length, dtype=torch.long)

        # Mappings
        query_mapping = torch.zeros(1, self.max_seq_length, dtype=torch.float)
        para_start_mapping = torch.zeros(1, self.para_limit, self.max_seq_length, dtype=torch.float)
        para_end_mapping = torch.zeros(1, self.para_limit, self.max_seq_length, dtype=torch.float)
        para_mapping = torch.zeros(1, self.max_seq_length, self.para_limit, dtype=torch.float)

        sent_start_mapping = torch.zeros(1, self.sent_limit, self.max_seq_length, dtype=torch.float)
        sent_end_mapping = torch.zeros(1, self.sent_limit, self.max_seq_length, dtype=torch.float)
        sent_mapping = torch.zeros(1, self.max_seq_length, self.sent_limit, dtype=torch.float)

        ent_start_mapping = torch.zeros(1, self.ent_limit, self.max_seq_length, dtype=torch.float)
        ent_end_mapping = torch.zeros(1, self.ent_limit, self.max_seq_length, dtype=torch.float)
        ent_mapping = torch.zeros(1, self.max_seq_length, self.ent_limit, dtype=torch.float)

        # Mask
        para_mask = torch.zeros(1, self.para_limit, dtype=torch.float)
        sent_mask = torch.zeros(1, self.sent_limit, dtype=torch.float)
        ent_mask = torch.zeros(1, self.ent_limit, dtype=torch.float)
        ans_cand_mask = torch.zeros(1, self.ent_limit, dtype=torch.float)

        # Label tensor
        y1 = torch.zeros(1, dtype=torch.long)
        y2 = torch.zeros(1, dtype=torch.long)
        q_type = torch.zeros(1, dtype=torch.long)
        is_support = torch.zeros(1, self.sent_limit, dtype=torch.float)
        is_gold_para = torch.zeros(1, self.para_limit, dtype=torch.float)
        is_gold_ent = torch.zeros(1, dtype=torch.float)

        # Graph related
        graphs = torch.zeros(1, self.graph_nodes_num, self.graph_nodes_num, dtype=torch.float)
        is_support.fill_(IGNORE_INDEX)
        is_gold_para.fill_(IGNORE_INDEX)
        is_gold_ent.fill_(IGNORE_INDEX)
        ################################################################################################################
        case = self.features[idx]
        ################################################################################################################
        i = 0
        context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
        context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
        segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

        if len(case.sent_spans) > 0:
            for j in range(case.sent_spans[0][0] - 1):
                query_mapping[i, j] = 1

        for j, para_span in enumerate(case.para_spans[:self.para_limit]):
            is_gold_flag = j in case.sup_para_ids
            start, end, _ = para_span
            if start <= end:
                end = min(end, self.max_seq_length - 1)
                is_gold_para[i, j] = int(is_gold_flag)
                para_mapping[i, start:end + 1, j] = 1
                para_start_mapping[i, j, start] = 1
                para_end_mapping[i, j, end] = 1

        for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
            is_sp_flag = j in case.sup_fact_ids
            start, end = sent_span
            if start <= end:
                end = min(end, self.max_seq_length - 1)
                is_support[i, j] = int(is_sp_flag)
                sent_mapping[i, start:end + 1, j] = 1
                sent_start_mapping[i, j, start] = 1
                sent_end_mapping[i, j, end] = 1

        for j, ent_span in enumerate(case.entity_spans[:self.ent_limit]):
            start, end = ent_span
            if start <= end:
                end = min(end, self.max_seq_length - 1)
                ent_mapping[i, start:end + 1, j] = 1
                ent_start_mapping[i, j, start] = 1
                ent_end_mapping[i, j, end] = 1
            ans_cand_mask[i, j] = int(j in case.answer_candidates_ids)

        is_gold_ent[i] = case.answer_in_entity_ids[0] if len(case.answer_in_entity_ids) > 0 else IGNORE_INDEX ## no need for loss computation

        if case.ans_type == 0 or case.ans_type == 3:
            if len(case.end_position) == 0:
                y1[i] = y2[i] = 0
            elif case.end_position[0] < self.max_seq_length and context_mask[i][
                case.end_position[0] + 1] == 1:  # "[SEP]" is the last token
                y1[i] = case.start_position[0]
                y2[i] = case.end_position[0]
            else:
                y1[i] = y2[i] = 0
            q_type[i] = case.ans_type if is_gold_ent[i] > 0 else 0
        elif case.ans_type == 1:
            y1[i] = IGNORE_INDEX
            y2[i] = IGNORE_INDEX
            q_type[i] = 1
        elif case.ans_type == 2:
            y1[i] = IGNORE_INDEX
            y2[i] = IGNORE_INDEX
            q_type[i] = 2
        # ignore entity loss if there is no entity
        if case.ans_type != 3:
            is_gold_ent[i].fill_(IGNORE_INDEX)

        tmp_graph = self.graph_dict[case.qas_id]
        graph_adj = torch.from_numpy(tmp_graph['adj'])
        for k in range(graph_adj.size(0)):
            graph_adj[k, k] = 8
        for edge_type in self.mask_edge_types:
            graph_adj = torch.where(graph_adj == edge_type, torch.zeros_like(graph_adj), graph_adj)
        graphs[i] = graph_adj

        id = case.qas_id
        input_length = (context_mask > 0).long().sum(dim=1)
        para_mask = (para_mapping > 0).any(1).float() ### 1 represents dimension
        sent_mask = (sent_mapping > 0).any(1).float()
        ent_mask = (ent_mapping > 0).any(1).float()

        res = {
            'context_idxs': context_idxs,
            'context_mask': context_mask,
            'segment_idxs': segment_idxs,
            'context_lens': input_length,
            'y1': y1,
            'y2': y2,
            'ids': id,
            'q_type': q_type,
            'is_support': is_support,
            'is_gold_para': is_gold_para,
            'is_gold_ent': is_gold_ent,
            'query_mapping': query_mapping,
            'para_mapping': para_mapping,
            'para_start_mapping': para_start_mapping,
            'para_end_mapping': para_end_mapping,
            'para_mask': para_mask,
            'sent_mapping': sent_mapping,
            'sent_start_mapping': sent_start_mapping,
            'sent_end_mapping': sent_end_mapping,
            'sent_mask': sent_mask,
            'ent_mapping': ent_mapping,
            'ent_start_mapping': ent_start_mapping,
            'ent_end_mapping': ent_end_mapping,
            'ent_mask': ent_mask,
            'ans_cand_mask': ans_cand_mask,
            'graphs': graphs
        }
        return res ## 26 elements

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 26
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
                batch_data[key] = torch.cat([_[key] for _ in data], dim=0)
        trim_keys = ['context_idxs', 'context_mask', 'segment_idxs', 'query_mapping']
        for key in trim_keys:
            batch_data[key] = batch_data[key][:,:max_c_len]
        trim_map_keys = ['para_mapping', 'sent_mapping', 'ent_mapping']
        for key in trim_map_keys:
            batch_data[key] = batch_data[key][:,:max_c_len,:]
        trim_start_end_keys = ['para_start_mapping', 'para_end_mapping',
                               'sent_start_mapping', 'sent_end_mapping',
                               'ent_start_mapping', 'ent_end_mapping']
        for key in trim_start_end_keys:
            batch_data[key] = batch_data[key][:, :, :max_c_len]
        return batch_data