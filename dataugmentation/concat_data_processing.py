import gzip
import pickle
import torch
import numpy as np

from os.path import join
from torch.utils.data import DataLoader

from envs import DATASET_FOLDER
from plmodels.pldata_processing import HotpotDataset, get_cached_filename

IGNORE_INDEX = -100

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AUG_MODULE_DICT = {'0': ('hgn'), '1': ('long'), '2': ('hgn', 'long'),
                   '3': ('hgn', 'hgn_reverse'), '4': ('long', 'long_reverse'),
                   '5': ('hgn', 'hgn_reverse', 'long', 'long_reverse')}
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class DataHelper:
    def __init__(self, gz=True, config=None):
        self.Dataset = HotpotDataset
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')
        self.config = config
        if self.config.daug_type not in AUG_MODULE_DICT:
            self.train_augf_types = AUG_MODULE_DICT['0']
        else:
            self.train_augf_types = AUG_MODULE_DICT[self.config.daug_type]
        self.devf_type = self.config.devf_type


    def get_feature_file(self, tag, f_type):
        cached_filename = get_cached_filename('{}_features'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_example_file(self, tag, f_type):
        cached_filename = get_cached_filename('{}_examples'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    def get_graph_file(self, tag, f_type):
        cached_filename = get_cached_filename('{}_graphs'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    # #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_feature_files(self, tag, f_types):
        cached_filenames = [(x, self.get_feature_file(tag=tag, f_type=x)) for x in f_types]
        return cached_filenames
    def get_example_files(self, tag, f_types):
        cached_filenames = [(x, self.get_example_file(tag=tag, f_type=x)) for x in f_types]
        return cached_filenames
    def get_graph_files(self, tag, f_types):
        cached_filenames = [(x, self.get_graph_file(tag=tag, f_type=x)) for x in f_types]
        return cached_filenames
    # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    # def train_feature_files(self):
    #     return self.get_feature_files('train', self.train_augf_types)
    #
    # def train_example_files(self):
    #     return self.get_example_files('train', self.train_augf_types)
    #
    # def train_graph_files(self):
    #     return self.get_graph_files('train', self.train_augf_types)
    #
    # def dev_feature_file(self):
    #     return self.get_feature_file('dev_distractor', self.config.devf_type)
    #
    # def dev_example_file(self):
    #     return self.get_example_file('dev_distractor', self.config.devf_type)
    #
    # def dev_graph_file(self):
    #     return self.get_graph_file('dev_distractor', self.config.devf_type)
    ###################################################################################################
    def get_pickle_file(self, file_name):
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def get_or_load(self, file):
        with self.get_pickle_file(file) as fin:
            print('loading', file)
            return pickle.load(fin)
    ###################################################################################################
    # Features
    def get_train_features(self):
        cached_features_names = self.get_feature_files(tag='train', f_types=self.train_augf_types)

        return

    def get_train_examples(self):
        cached_examples_names = self.get_example_files(tag='train', f_types=self.train_augf_types)
        return

    def get_train_graphs(self):
        cached_graph_names = self.get_graph_files(tag='train')
        return

    def get_dev_features(self):
        return

    def get_dev_examples(self):
        return

    def get_dev_graphs(self):
        return

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