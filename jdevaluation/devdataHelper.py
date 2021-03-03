from __future__ import absolute_import, division, print_function
from os.path import join
from torch.utils.data import DataLoader
from plmodels.pldata_processing import HotpotDataset, get_cached_filename
from envs import DATASET_FOLDER
import gzip
import pickle

class DataHelper:
    def __init__(self, gz=True, config=None):
        self.Dataset = HotpotDataset
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')
        self.__dev_features__ = None
        self.__dev_examples__ = None
        self.__dev_graphs__ = None
        self.__dev_example_dict__ = None

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
    def dev_feature_file(self):
        return self.get_feature_file('dev_distractor', self.config.devf_type)

    @property
    def dev_example_file(self):
        return self.get_example_file('dev_distractor', self.config.devf_type)

    @property
    def dev_graph_file(self):
        return self.get_graph_file('dev_distractor', self.config.devf_type)

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
    def dev_features(self):
        return self.__get_or_load__('__dev_features__', self.dev_feature_file)

    # Examples
    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)

    # Graphs
    @property
    def dev_graphs(self):
        return self.__get_or_load__('__dev_graphs__', self.dev_graph_file)

    # Example dict
    @property
    def dev_example_dict(self):
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    # Feature dict
    @property
    def dev_feature_dict(self):
        return {e.qas_id: e for e in self.dev_features}

    # Load
    def load_dev(self):
        return self.dev_features, self.dev_example_dict, self.dev_graphs

    @property
    def dev_loader(self):
        return self.Dataset(*self.load_dev(),
                                 para_limit=self.config.max_para_num,
                                 sent_limit=self.config.max_sent_num,
                                 ent_limit=self.config.max_entity_num,
                                 ans_ent_limit=self.config.max_ans_ent_num,
                                 mask_edge_types=self.config.mask_edge_types)
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