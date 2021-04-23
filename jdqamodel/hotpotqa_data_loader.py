from torch.utils.data import Dataset
from jdqamodel.hotpotqa_data_structure import Example
from jdqamodel.hotpotqaUtils import example_sent_drop, case_to_features, trim_input_span
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
            doc_input_ids, query_spans, para_spans, sent_spans, limit=512, sep_token_id=self.sep_token_id, ans_spans=ans_spans)


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
            limit=512, sep_token_id=self.sep_token_id, ans_spans=ans_spans)

class HotpotTestDataset(Dataset):
    def __init__(self, examples, sep_token_id, max_para_num=4, max_sent_num=60,
                 max_seq_num=512):
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
            limit=512, sep_token_id=self.sep_token_id)