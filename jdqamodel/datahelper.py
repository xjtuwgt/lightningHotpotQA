from jdqamodel.hotpotqa_data_loader import HotpotDataset
from torch.utils.data import DataLoader
from os.path import join
from envs import DATASET_FOLDER
import gzip, pickle
from jdqamodel.hotpotqa_dump_features import get_cached_filename
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DataHelper:
    def __init__(self, sep_token_id, gz=True, config=None):
        self.Dataset = HotpotDataset
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')
        self.config = config
        self.train_examples = None
        self.train_example_dict = None
        self.dev_examples = None
        self.dev_example_dict = None
        self.sep_token_id = sep_token_id
        self.train_f_type = self.config.daug_type

    def get_example_file(self, tag, f_type=None):
        cached_filename = get_cached_filename('{}_hotpotqa_tokenized_examples'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    def train_example_file(self):
        return self.get_example_file('train', self.train_f_type)

    def dev_example_file(self):
        return self.get_example_file('dev_distractor', self.config.devf_type)

    def get_pickle_file(self, file_name):
        print('pickler file name {}'.format(file_name))
        return pickle.load(gzip.open(file_name, 'rb'))

    # Examples
    def get_train_examples(self):
        return self.get_pickle_file(self.train_example_file())

    def get_dev_examples(self):
        return self.get_pickle_file(self.dev_example_file())

    def hotpot_train_dataloader(self) -> DataLoader:
        self.train_examples = self.get_train_examples()
        self.train_example_dict = {e.qas_id: e for e in self.train_examples}
        train_data = self.Dataset(examples=self.train_examples,
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

    def hotpot_val_dataloader(self) -> DataLoader:
        self.dev_examples = self.get_dev_examples()
        self.dev_example_dict = {e.qas_id: e for e in self.dev_examples}
        dev_data = self.Dataset(examples=self.dev_examples,
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
            collate_fn=HotpotDataset.collate_fn)
        return dataloader