import json
import pandas as pd
from pandas import DataFrame
from time import time
from tqdm import tqdm

def json_loader(json_file_name: str):
    with open(json_file_name, 'r', encoding='utf-8') as reader:
        json_data = json.load(reader)
    return json_data

def loadWikiData(json_file_name: str)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(json_file_name, orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

#########+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    return question

def normalize_text(text: str) -> str:
    text = ' ' + text.lower().strip() ###adding the ' ' is important to make the consist encoder, for roberta tokenizer
    return text

def answer_span_checker(answer, sentence):
    find_idx = sentence.find(answer)
    if find_idx < 0:
        return False
    return True

def find_answer_span(norm_answer, sentence, tokenizer):
    answer_encode_ids = tokenizer.text_encode(text=norm_answer, add_special_tokens=False)
    sentence_encode_ids = tokenizer.text_encode(text=sentence, add_special_tokens=False)
    idx = sub_list_match_idx(target=answer_encode_ids, source=sentence_encode_ids)
    flag = idx >= 0
    return flag, answer_encode_ids, sentence_encode_ids

def find_sub_list_fuzzy(target: list, source: list) -> int:
    if len(target) > len(source):
        return -1
    t_len = len(target)
    temp_idx = -1
    if t_len >=4:
        temp_target = target[1:]
        temp_idx = find_sub_list(temp_target, source)
        if temp_idx < 1:
            temp_target = target[:(t_len-1)]
            temp_idx = find_sub_list(temp_target, source)
        else:
            temp_idx = temp_idx - 1
    return temp_idx

def sub_list_match_idx(target: list, source: list) -> int:
    idx = find_sub_list(target, source)
    if idx < 0:
        idx = find_sub_list_fuzzy(target, source)
    return idx

def find_sub_list(target: list, source: list) -> int:
    if len(target) > len(source):
        return -1
    t_len = len(target)
    def equal_list(a_list, b_list):
        for j in range(len(a_list)):
            if a_list[j] != b_list[j]:
                return False
        return True
    for i in range(len(source) - len(target) + 1):
        temp = source[i:(i+t_len)]
        is_equal = equal_list(target, temp)
        if is_equal:
            return i
    return -1
########################################################################################################################

def pos_neg_context_split(row, tokenizer):
    question, supporting_facts, contexts, answer = row['question'], row['supporting_facts'], row['context'], row['answer']
    doc_title2doc_len = dict([(title, len(text)) for title, text in contexts])
    supporting_facts_filtered = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                 if supp_sent_idx < doc_title2doc_len[supp_title]] ##some supporting facts are out of sentence index
    positive_titles = set([x[0] for x in supporting_facts_filtered]) ## get postive document titles
    norm_answer = normalize_text(text=answer) ## normalize the answer (add a space between the answer)
    norm_question = normalize_question(question.lower()) ## normalize the question by removing the question mark
    not_found_flag = False ## some answer might be not founded in supporting sentence
    ################################################################################################################
    pos_doc_num = len(positive_titles) ## number of positive documents
    pos_ctxs, neg_ctxs = [], []
    for ctx_idx, ctx in enumerate(contexts): ## Original ctx index, record the original index order
        title, text = ctx[0], ctx[1]
        text_lower = [normalize_text(text=sent) for sent in text]
        if title in positive_titles:
            count = 1
            supp_sent_flags = []
            for supp_title, supp_sent_idx in supporting_facts_filtered:
                if title == supp_title:
                    supp_sent = text_lower[supp_sent_idx]
                    if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
                        has_answer = answer_span_checker(norm_answer.strip(), supp_sent)
                        if has_answer:
                            encode_has_answer, X, Y = find_answer_span(norm_answer.strip(), supp_sent, tokenizer)
                            if not encode_has_answer:
                                encode_has_answer, X, Y = find_answer_span(norm_answer, supp_sent, tokenizer)
                                if not encode_has_answer:
                                    supp_sent_flags.append((supp_sent_idx, False))
                                else:
                                    supp_sent_flags.append((supp_sent_idx, True))
                                    count = count + 1
                            else:
                                supp_sent_flags.append((supp_sent_idx, True))
                                count = count + 1
                        else:
                            supp_sent_flags.append((supp_sent_idx, False))
                    else:
                        supp_sent_flags.append((supp_sent_idx, False))
            pos_ctxs.append([title.lower(), text_lower, count, supp_sent_flags, ctx_idx])  ## Identify the support document with answer
        else:
            neg_ctxs.append([title.lower(), text_lower, 0, [], ctx_idx])
    neg_doc_num = len(neg_ctxs)
    pos_counts = [x[2] for x in pos_ctxs]
    if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
        if sum(pos_counts) == 2:
            not_found_flag = True
    assert len(pos_counts) == 2
    if not_found_flag:
        norm_answer = 'noanswer'
    if (pos_counts[0] > 1 and pos_counts[1] > 1) or (pos_counts[0] <= 1 and pos_counts[1] <= 1):
        answer_type = False
    else:
        answer_type = True
    yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
    return norm_question, norm_answer, pos_ctxs, neg_ctxs, supporting_facts_filtered, answer_type, pos_doc_num, neg_doc_num, yes_no_flag, not_found_flag

def hotpot_answer_tokenizer(para_file, full_file, tokenizer):
    sel_para_data = json_loader(json_file_name=para_file)
    full_data = json_loader(json_file_name=full_file)

    examples = []
    for row in tqdm(full_data):
        key = row['_id']
