from torch.utils.data import Dataset
from jdqamodel.hotpotqa_data_structure import Example
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
        if self.sent_drop_ratio > 0:
            case = example_sent_drop(case=case, drop_ratio=self.sent_drop_ratio)
        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = case_to_features(case=case, train_dev=True)


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
        doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, ans_type_label = case_to_features(case=case,
                                                                                         train_dev=True)
#######################################################################
def case_to_features(case: Example, train_dev=True):
    question_input_ids = case.question_input_ids
    ctx_input_ids = case.ctx_input_ids
    sent_num = case.sent_num
    para_num = case.para_num
    para_names = case.para_names
    sent_names = case.sent_names
    assert len(ctx_input_ids) == para_num and sent_num == len(sent_names)
    doc_input_ids = [] ### ++++++++
    doc_input_ids += question_input_ids
    para_len_list = [len(question_input_ids)]
    sent_len_list = [len(question_input_ids)]
    query_len = len(question_input_ids)
    query_spans = [(1, query_len)]
    para_sent_pair_to_sent_id, sent_id = {}, 0
    for para_idx, para_name in enumerate(para_names):
        para_sent_ids = ctx_input_ids[para_idx]
        para_len_ = 0
        for sent_idx, sent_ids in enumerate(para_sent_ids):
            doc_input_ids += sent_ids
            sent_len_i = len(sent_ids)
            sent_len_list.append(sent_len_i)
            para_len_ = para_len_ + sent_len_i
            para_sent_pair_to_sent_id[(para_name, sent_idx)] = sent_id
            sent_id = sent_id + 1
        para_len_list.append(para_len_)
    # print('In here {}'.format(doc_input_ids))
    assert sent_num == len(sent_len_list) - 1 and para_num == len(para_len_list) - 1
    assert sent_id == sent_num
    sent_cum_sum_len_list = np.cumsum(sent_len_list).tolist()
    para_cum_sum_len_list = np.cumsum(para_len_list).tolist()
    sent_spans = [(sent_cum_sum_len_list[i], sent_cum_sum_len_list[i+1]) for i in range(sent_num)]
    para_spans = [(para_cum_sum_len_list[i], para_cum_sum_len_list[i+1]) for i in range(para_num)]
    assert len(sent_spans) == sent_num
    assert len(para_spans) == para_num
    if train_dev:
        answer_text = case.answer_text.strip()
        if answer_text in ['yes']:
            answer_type_label = [0]
        elif answer_text in ['no', 'noanswer']:
            answer_type_label = [1]
        else:
            answer_type_label = [2]
        answer_positions = case.answer_positions
        ans_spans = []
        for ans_position in answer_positions:
            doc_title, sent_id, ans_start, ans_end = ans_position
            sent_idx = para_sent_pair_to_sent_id[(doc_title, sent_id)]
            sent_start_idx = sent_spans[sent_idx][0]
            ans_spans.append((sent_start_idx + ans_start, sent_start_idx + ans_end))

        # print('in', len(doc_input_ids))
        return doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, answer_type_label
    else:
        return doc_input_ids, query_spans, para_spans, sent_spans
#######################################################################
def largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx
    return len(spans)

def trim_input_span(doc_input_ids, query_spans, para_spans, sent_spans, limit, sep_token_id, ans_spans=None):
    if len(doc_input_ids) <= limit:
        if ans_spans is not None:
            return doc_input_ids, query_spans, para_spans, sent_spans, ans_spans
        else:
            return doc_input_ids, query_spans, para_spans, sent_spans
    else:
        trim_doc_input_ids = []
        trim_doc_input_ids += doc_input_ids[:(limit-1)]
        trim_doc_input_ids += [sep_token_id]
        largest_para_idx = largest_valid_index(para_spans, limit)
        trim_para_spans = []
        trim_para_spans += para_spans[:(largest_para_idx+1)]
        trim_para_spans = [[_[0], _[1]] for _ in trim_para_spans]
        trim_para_spans[largest_para_idx][1] = limit
        trim_para_spans = [(_[0], _[1]) for _ in trim_para_spans]

        largest_sent_idx = largest_valid_index(sent_spans, limit)
        trim_sent_spans = []
        trim_sent_spans += sent_spans[:(largest_sent_idx+1)]
        trim_sent_spans = [[_[0], _[1]] for _ in trim_sent_spans]
        trim_sent_spans[largest_sent_idx][1] = limit
        trim_sent_spans = [(_[0], _[1]) for _ in trim_sent_spans]

        # print('Para Trim here, {}\n {}'.format(trim_para_spans, para_spans))
        # print('Sent Trim here, {}\n {}'.format(trim_sent_spans, sent_spans))


        if ans_spans is not None:
            largest_ans_idx = largest_valid_index(ans_spans, limit)
            print('largest idx {}\t{}\t{}'.format(largest_ans_idx, len(ans_spans), ans_spans))
            trim_ans_spans = []
            trim_ans_spans += ans_spans[:largest_ans_idx]
            if largest_ans_idx == 0:
                print('trim ans hhhhhhhhhhhhhhhhhhh')
            return trim_doc_input_ids, query_spans, trim_para_spans, trim_sent_spans, trim_ans_spans
        else:
            return trim_doc_input_ids, query_spans, trim_para_spans, trim_sent_spans

#######################################################################
def example_sent_drop(case: Example, drop_ratio:float = 0.1):
    qas_id = case.qas_id
    qas_type = case.qas_type
    question_tokens = case.question_tokens
    ctx_tokens = case.ctx_tokens
    question_text = case.question_text
    question_input_ids = case.question_input_ids
    ctx_input_ids = case.ctx_input_ids
    sent_names = case.sent_names
    para_names = case.para_names
    sup_fact_id = case.sup_fact_id
    sup_para_id = case.sup_para_id
    ctx_text = case.ctx_text
    answer_text = case.answer_text
    answer_tokens = case.answer_tokens
    answer_input_ids = case.answer_input_ids
    answer_positions = case.answer_positions
    ctx_with_answer = case.ctx_with_answer
    para_num = case.para_num
    sent_num = case.sent_num
    assert para_num == len(ctx_tokens) and para_num == len(ctx_input_ids) and para_num == len(para_names)
    assert sent_num == len(sent_names)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sent_name_to_id_dict = dict([((x[1][0], x[1][1]), x[0]) for x in enumerate(sent_names)])
    keep_sent_idxs = []
    drop_ctx_tokens = []
    drop_ctx_input_ids = []
    for para_idx, para_name in enumerate(para_names):
        drop_para_ctx_tokens, drop_para_input_ids = [], []
        for sent_idx, (sent_sub_token, sent_inp_ids) in enumerate(zip(ctx_tokens[para_idx], ctx_input_ids[para_idx])):
            abs_sent_idx = sent_name_to_id_dict[(para_name, sent_idx)]
            if abs_sent_idx not in sup_fact_id:
                rand_s_i = random.rand()
                if rand_s_i > drop_ratio:
                    drop_para_ctx_tokens.append(sent_sub_token)
                    drop_para_input_ids.append(sent_inp_ids)
                    keep_sent_idxs.append(abs_sent_idx)
            else:
                drop_para_ctx_tokens.append(sent_sub_token)
                drop_para_input_ids.append(sent_inp_ids)
                keep_sent_idxs.append(abs_sent_idx)
        drop_ctx_tokens.append(drop_para_ctx_tokens)
        drop_ctx_input_ids.append(drop_para_input_ids)
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    keep_sent_idx_remap_dict = dict([(x[1], x[0]) for x in enumerate(keep_sent_idxs)]) ## for answer map
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    drop_para_names = []
    for para_idx, para_name in enumerate(para_names):
        assert len(drop_ctx_input_ids[para_idx]) == len(drop_ctx_tokens[para_idx])
        if len(drop_ctx_tokens[para_idx]) > 0:
            drop_para_names.append(para_name)
    drop_ctx_tokens = [_ for _ in drop_ctx_tokens if len(_) > 0]
    drop_ctx_input_ids = [_ for _ in drop_ctx_input_ids if len(_) > 0]
    drop_para_fact_id = []
    supp_para_names = [para_names[_] for _ in sup_para_id]
    for para_idx, para_name in enumerate(drop_para_names):
        if para_name in supp_para_names:
            drop_para_fact_id.append(para_idx)
    drop_para_num = len(drop_para_names)
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    drop_sent_names = []
    for para_idx, para_name in enumerate(drop_para_names):
        for sent_idx in range(len(drop_ctx_input_ids[para_idx])):
            drop_sent_names.append((para_name, sent_idx))
    drop_supp_fact_ids = [keep_sent_idx_remap_dict[_] for _ in sup_fact_id]
    drop_sent_num = len(drop_sent_names)
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    drop_answer_positions = []
    for answer_position in answer_positions:
        title, sent_idx, start_pos, end_pos = answer_position
        orig_sent_name = (title, sent_idx)
        orig_abs_sent_idx = sent_name_to_id_dict[orig_sent_name]
        drop_abs_sent_idx = keep_sent_idx_remap_dict[orig_abs_sent_idx]
        drop_sent_name = drop_sent_names[drop_abs_sent_idx]
        assert drop_sent_name[0] == title
        drop_answer_positions.append((drop_sent_name[0], drop_sent_name[1], start_pos, end_pos))

    drop_example = Example(
        qas_id=qas_id,
        qas_type=qas_type,
        ctx_text=ctx_text,
        ctx_tokens=drop_ctx_tokens,
        ctx_input_ids=drop_ctx_input_ids,
        para_names=drop_para_names,
        sup_para_id=drop_para_fact_id,
        sent_names=drop_sent_names,
        para_num=drop_para_num,
        sent_num=drop_sent_num,
        sup_fact_id=drop_supp_fact_ids,
        question_text=question_text,
        question_tokens=question_tokens,
        question_input_ids=question_input_ids,
        answer_text=answer_text,
        answer_tokens=answer_tokens,
        answer_input_ids=answer_input_ids,
        answer_positions=drop_answer_positions,
        ctx_with_answer=ctx_with_answer)
    return drop_example

class HotpotTestDataset(Dataset):
    def __init__(self, examples, max_para_num=4, max_sent_num=60,
                 max_seq_num=512):
        self.examples = examples
        self.max_para_num = max_para_num
        self.max_sent_num = max_sent_num
        self.max_seq_length = max_seq_num

    def __getitem__(self, idx):
        case: Example = self.examples[idx]
        doc_input_ids, query_spans, para_spans, sent_spans = case_to_features(case=case, train_dev=False)

