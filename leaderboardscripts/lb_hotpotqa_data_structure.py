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
    f_type_set = {'examples', 'features', 'graphs',
                  'hgn_examples', 'hgn_features', 'hgn_graphs',
                  'hgn_reverse_examples', 'hgn_reverse_features', 'hgn_reverse_graphs',

                  'hgn_low_examples', 'hgn_low_features', 'hgn_low_graphs',
                  'hgn_low_reverse_examples', 'hgn_low_reverse_features', 'hgn_low_reverse_graphs',

                  'hgn_low_sae_examples', 'hgn_low_sae_features', 'hgn_low_sae_graphs',
                  'hgn_low_sae_reverse_examples', 'hgn_low_sae_reverse_features', 'hgn_low_sae_reverse_graphs',

                  'long_examples', 'long_features', 'long_graphs',
                  'long_reverse_examples', 'long_reverse_features', 'long_reverse_graphs',

                  'long_low_examples', 'long_low_features', 'long_low_graphs',
                  'long_low_reverse_examples', 'long_low_reverse_features', 'long_low_reverse_graphs',

                  'long_low_sae_examples', 'long_low_sae_features', 'long_low_sae_graphs',
                  'long_low_sae_reverse_examples', 'long_low_sae_reverse_features', 'long_low_sae_reverse_graphs',

                  'docred_low_examples', 'docred_low_features', 'docred_low_graphs',
                  'docred_low_sae_examples', 'docred_low_sae_features', 'docred_low_sae_graphs',

                  'hgn_docred_low_examples', 'hgn_docred_low_features', 'hgn_docred_low_graphs',
                  'hgn_docred_low_sae_examples', 'hgn_docred_low_sae_features', 'hgn_docred_low_sae_graphs',

                  'long_docred_low_examples', 'long_docred_low_features', 'long_docred_low_graphs',
                  'long_docred_low_sae_examples', 'long_docred_low_sae_features', 'long_docred_low_sae_graphs',

                  'hgn_long_docred_low_examples', 'hgn_long_docred_low_features', 'hgn_long_docred_low_graphs',
                  'hgn_long_docred_low_sae_examples', 'hgn_long_docred_low_sae_features', 'hgn_long_docred_low_sae_graphs',

                  'hgn_long_low_examples', 'hgn_long_low_features', 'hgn_long_low_graphs',
                  'hgn_long_low_sae_examples', 'hgn_long_low_sae_features', 'hgn_long_low_sae_graphs',

                  'oracle_features', 'oracle_graphs', 'oracle_examples',
                  'oracle_sae_features', 'oracle_sae_graphs', 'oracle_sae_examples',

                  'oracle_features', 'oracle_graphs', 'oracle_examples',
                  'oracle_sae_features', 'oracle_sae_graphs', 'oracle_sae_examples',

                  'hgn_low_reranker2_examples', 'hgn_low_reranker2_features', 'hgn_low_reranker2_graphs',
                  'hgn_low_reranker3_examples', 'hgn_low_reranker3_features', 'hgn_low_reranker3_graphs',

                  'hgn_low_sae_reranker2_examples', 'hgn_low_sae_reranker2_features', 'hgn_low_sae_reranker2_graphs',
                  'hgn_low_sae_reranker3_examples', 'hgn_low_sae_reranker3_features', 'hgn_low_sae_reranker3_graphs',

                  'long_low_reranker2_examples', 'long_low_reranker2_features', 'long_low_reranker2_graphs',
                  'long_low_reranker3_examples', 'long_low_reranker3_features', 'long_low_reranker3_graphs',

                  'long_low_sae_reranker2_examples', 'long_low_sae_reranker2_features', 'long_low_sae_reranker2_graphs',
                  'long_low_sae_reranker3_examples', 'long_low_sae_reranker3_features', 'long_low_sae_reranker3_graphs',
                  #++++

                  'roberta_hgn_low_reranker2_examples', 'roberta_hgn_low_reranker2_features', 'roberta_hgn_low_reranker2_graphs',
                  'roberta_hgn_low_reranker3_examples', 'roberta_hgn_low_reranker3_features', 'roberta_hgn_low_reranker3_graphs',

                  'roberta_hgn_low_sae_reranker2_examples', 'roberta_hgn_low_sae_reranker2_features', 'roberta_hgn_low_sae_reranker2_graphs',
                  'roberta_hgn_low_sae_reranker3_examples', 'roberta_hgn_low_sae_reranker3_features', 'roberta_hgn_low_sae_reranker3_graphs',

                  'albert_long_low_reranker2_examples', 'albert_long_low_reranker2_features', 'albert_long_low_reranker2_graphs',
                  'albert_long_low_reranker3_examples', 'albert_long_low_reranker3_features', 'albert_long_low_reranker3_graphs',

                  'albert_long_low_sae_reranker2_examples', 'albert_long_low_sae_reranker2_features', 'albert_long_low_sae_reranker2_graphs',
                  'albert_long_low_sae_reranker3_examples', 'albert_long_low_sae_reranker3_features', 'albert_long_low_sae_reranker3_graphs',

                  } #### ranker: hgn, longformer; case: lowercase, cased; graph: whether sae-graph
    assert f_type in f_type_set
    return f"cached_{f_type}_{config.model_type}_{config.max_seq_length}_{config.max_query_length}.pkl.gz"

class Example(object):

    def __init__(self,
                 qas_id,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 para_names,
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
                 edges=None):
        self.qas_id = qas_id
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.para_names = para_names
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
                 token_to_orig_map,
                 edges=None):
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

        self.edges = edges
        self.token_to_orig_map = token_to_orig_map


class DataHelper:
    def __init__(self, gz=True, config=None):
        self.Dataset = HotpotDataset
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')
        self.__test_features__ = None
        self.__test_examples__ = None
        self.__test_graphs__ = None
        self.__test_example_dict__ = None

        self.config = config

    def get_feature_file(self, tag, f_type=None):
        cached_filename = get_cached_filename('{}_features'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_example_file(self, tag, f_type=None):
        cached_filename = get_cached_filename('{}_examples'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_graph_file(self, tag, f_type=None):
        cached_filename = get_cached_filename('{}_graphs'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)


    @property
    def test_feature_file(self):
        return self.get_feature_file('test_distractor', self.config.testf_type)

    @property
    def test_example_file(self):
        return self.get_example_file('test_distractor', self.config.testf_type)

    @property
    def test_graph_file(self):
        return self.get_graph_file('test_distractor', self.config.testf_type)

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
    def test_features(self):
        return self.__get_or_load__('__test_features__', self.test_feature_file)

    # Examples
    @property
    def test_examples(self):
        return self.__get_or_load__('__test_examples__', self.test_example_file)

    # Graphs
    @property
    def test_graphs(self):
        return self.__get_or_load__('__test_graphs__', self.test_graph_file)

    # Example dict
    @property
    def test_example_dict(self):
        if self.__test_example_dict__ is None:
            self.__test_example_dict__ = {e.qas_id: e for e in self.test_examples}
        return self.__test_example_dict__

    # Feature dict
    @property
    def test_feature_dict(self):
        return {e.qas_id: e for e in self.test_features}

    # Load
    def load_test(self):
        return self.test_features, self.test_example_dict, self.test_graphs

    @property
    def hotpot_test_dataloader(self) -> DataLoader:

        dev_data = self.Dataset(*self.load_test(),
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 num_edge_type=self.config.num_edge_type,
                                 mask_edge_types=self.config.mask_edge_types)

        dataloader = DataLoader(
            dataset=dev_data,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=HotpotDataset.collate_fn
        )
        return dataloader


class HotpotDataset(Dataset):
    def __init__(self, features, example_dict, graph_dict, para_limit, sent_limit, ent_limit,
                 mask_edge_types, num_edge_type):
        self.features = features
        self.example_dict = example_dict
        self.graph_dict = graph_dict
        self.para_limit = para_limit
        self.sent_limit = sent_limit
        self.ent_limit = ent_limit
        self.graph_nodes_num = 1 + para_limit + sent_limit + ent_limit
        self.mask_edge_types = mask_edge_types
        self.num_edge_type = num_edge_type
        self.max_seq_length = 512

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # BERT input
        context_idxs = torch.zeros(1, self.max_seq_length, dtype=torch.long)
        context_mask = torch.zeros(1, self.max_seq_length, dtype=torch.long)
        segment_idxs = torch.zeros(1, self.max_seq_length, dtype=torch.long)

        # # Mappings
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
        # ans_cand_mask = torch.zeros(1, self.ent_limit, dtype=torch.float)

        # Graph related
        graphs = torch.zeros(1, self.graph_nodes_num, self.graph_nodes_num, dtype=torch.float)
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
            start, end, _ = para_span
            if start <= end:
                end = min(end, self.max_seq_length - 1)
                # is_gold_para[i, j] = int(is_gold_flag)
                para_mapping[i, start:end + 1, j] = 1
                para_start_mapping[i, j, start] = 1
                para_end_mapping[i, j, end] = 1

        for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
            start, end = sent_span
            if start <= end:
                end = min(end, self.max_seq_length - 1)
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
        #     # ans_cand_mask[i, j] = int(j in case.answer_candidates_ids)
        #
        # # is_gold_ent[i] = case.answer_in_entity_ids[0] if len(case.answer_in_entity_ids) > 0 else IGNORE_INDEX ## no need for loss computation
        #
        # # if case.ans_type == 0 or case.ans_type == 3:
        # #     if len(case.end_position) == 0:
        # #         y1[i] = y2[i] = 0
        # #     elif case.end_position[0] < self.max_seq_length and context_mask[i][
        # #         case.end_position[0] + 1] == 1:  # "[SEP]" is the last token
        # #         y1[i] = case.start_position[0]
        # #         y2[i] = case.end_position[0]
        # #     else:
        # #         y1[i] = y2[i] = 0
        # #     q_type[i] = case.ans_type if is_gold_ent[i] > 0 else 0
        # # elif case.ans_type == 1:
        # #     y1[i] = IGNORE_INDEX
        # #     y2[i] = IGNORE_INDEX
        # #     q_type[i] = 1
        # # elif case.ans_type == 2:
        # #     y1[i] = IGNORE_INDEX
        # #     y2[i] = IGNORE_INDEX
        # #     q_type[i] = 2
        # # # ignore entity loss if there is no entity
        # # if case.ans_type != 3:
        # #     is_gold_ent[i].fill_(IGNORE_INDEX)
        #
        tmp_graph = self.graph_dict[case.qas_id]
        graph_adj = torch.from_numpy(tmp_graph['adj'])
        for k in range(graph_adj.size(0)):
            graph_adj[k, k] = self.num_edge_type ## adding self-loop
        for edge_type in self.mask_edge_types:
            graph_adj = torch.where(graph_adj == edge_type, torch.zeros_like(graph_adj), graph_adj)
        graphs[i] = graph_adj

        examp_id = case.qas_id
        input_length = (context_mask > 0).long().sum(dim=1)
        para_mask = (para_mapping > 0).any(1).float() ### 1 represents dimension
        sent_mask = (sent_mapping > 0).any(1).float()
        ent_mask = (ent_mapping > 0).any(1).float()
        #
        res = {
            'context_idxs': context_idxs,
            'context_mask': context_mask,
            'segment_idxs': segment_idxs,
            'context_lens': input_length,
            'ids': examp_id,
            # 'y1': y1,
            # 'y2': y2,
            # 'q_type': q_type,
            # 'is_support': is_support,
            # 'is_gold_para': is_gold_para,
            # 'is_gold_ent': is_gold_ent,
            # 'ans_cand_mask': ans_cand_mask,
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
            'graphs': graphs}
        # res = {'ids': examp_id,
        #        'context_idxs': context_idxs,
        #        'context_mask': context_mask,
        #        'segment_idxs': segment_idxs
        #        }
        return res ## 19 elements

    @staticmethod
    def collate_fn(data):
        # # assert len(data[0]) == 19
        context_lens_np = np.array([_['context_lens'] for _ in data])
        max_c_len = context_lens_np.max()
        sorted_idxs = np.argsort(context_lens_np)[::-1]
        assert len(data) == len(sorted_idxs)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        data = [data[_] for _ in sorted_idxs]
        # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        # trim_start_end_keys = ['para_start_mapping', 'para_end_mapping',
        #                        'sent_start_mapping', 'sent_end_mapping',
        #                        'ent_start_mapping', 'ent_end_mapping']
        # for key in trim_start_end_keys:
        #     batch_data[key] = batch_data[key][:, :, :max_c_len]

        return batch_data