import json
import pandas as pd
from pandas import DataFrame
from time import time
from tqdm import tqdm
import itertools
from HotpotQAModel.hotpotqa_data_structure import Example

def json_loader(json_file_name: str):
    with open(json_file_name, 'r', encoding='utf-8') as reader:
        json_data = json.load(reader)
    return json_data

def loadWikiData(json_file_name: str)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(json_file_name, orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
def selected_context_processing(row, tokenizer, selected_para_titles):
    question, supporting_facts, contexts, answer = row['question'], row['supporting_facts'], row['context'], row['answer']
    doc_title2doc_len = dict([(title, len(text)) for title, text in contexts])
    supporting_facts_filtered = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                 if supp_sent_idx < doc_title2doc_len[supp_title]] ##some supporting facts are out of sentence index
    positive_titles = set([x[0] for x in supporting_facts_filtered]) ## get postive document titles
    ################################################################################################################
    norm_answer = normalize_text(text=answer) ## normalize the answer (add a space between the answer)
    norm_question = normalize_question(question.lower()) ## normalize the question by removing the question mark
    ################################################################################################################
    answer_found_flag = False ## some answer might be not founded in supporting sentence
    ################################################################################################################
    selected_contexts = []
    context_dict = dict(row['context'])
    for title in selected_para_titles:
        para_text = context_dict[title]
        para_text_lower = [normalize_text(text=sent) for sent in para_text]
        if title in positive_titles:
            count = 1
            supp_sent_flags = []
            for supp_title, supp_sent_idx in supporting_facts_filtered:
                if title == supp_title:
                    supp_sent = para_text_lower[supp_sent_idx]
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
                                    answer_found_flag = True
                            else:
                                supp_sent_flags.append((supp_sent_idx, True))
                                count = count + 1
                                answer_found_flag = True
                        else:
                            supp_sent_flags.append((supp_sent_idx, False))
                    else:
                        supp_sent_flags.append((supp_sent_idx, False))
            selected_contexts.append([title, para_text_lower, count, supp_sent_flags, True])  ## support para
        else:
            selected_contexts.append([title, para_text_lower, 0, [], False]) ## no support para
    if not answer_found_flag:
        norm_answer = 'noanswer'
    yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
    return norm_question, norm_answer, selected_contexts, supporting_facts_filtered, yes_no_flag, answer_found_flag
#=======================================================================================================================
def hotpot_answer_tokenizer(para_file: str,
                            full_file: str,
                            tokenizer,
                            cls_token='[CLS]',
                            sep_token='[SEP]',
                            is_roberta=False):
    sel_para_data = json_loader(json_file_name=para_file)
    full_data = json_loader(json_file_name=full_file)
    examples = []
    answer_not_found_count = 0
    for row in tqdm(full_data):
        key = row['_id']
        qas_type = row['type']
        sent_names = []
        sup_facts_sent_id = []
        para_names = []
        sup_para_id = []
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sel_paras = sel_para_data[key]
        selected_para_titles = itertools.chain.from_iterable(sel_paras)
        norm_question, norm_answer, selected_contexts, supporting_facts_filtered, yes_no_flag, answer_found_flag = \
            selected_context_processing(row=row, tokenizer=tokenizer, selected_para_titles=selected_para_titles)
        if not answer_found_flag:
            answer_not_found_count = answer_not_found_count + 1
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_tokens = [cls_token]
        if is_roberta:
            sub_tokens = tokenizer.tokenize(norm_question, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(norm_question)
        query_tokens += sub_tokens
        if is_roberta:
            query_tokens += [sep_token, sep_token]
        else:
            query_tokens += [sep_token]
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        assert len(query_tokens) == len(query_input_ids)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_to_id, sent_id = {}, 0
        ctx_token_list = []
        ctx_input_id_list = []
        sent_num = 0
        para_num = 0
        for para_idx, para_tuple in enumerate(selected_contexts):
            para_num += 1
            title, sents, _, answer_sent_flags, supp_para_flag = para_tuple
            para_names.append(title)
            if supp_para_flag:
                sup_para_id.append(para_idx)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_tokens_list = []
            sent_input_id_list = []
            for local_sent_id, sent_text in enumerate(sents):
                sent_num += 1
                local_sent_name = (title, local_sent_id)
                sent_to_id[local_sent_name] = sent_id
                sent_names.append(local_sent_name)
                if local_sent_name in supporting_facts_filtered:
                    sup_facts_sent_id.append(sent_id)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if is_roberta:
                    sub_tokens = tokenizer.tokenize(sent_text, add_prefix_space=True)
                    sub_tokens.append(sep_token)
                else:
                    sub_tokens = tokenizer.tokenize(sent_text)
                sub_input_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                assert len(sub_input_ids) == len(sub_tokens)
                sent_tokens_list.append(sub_tokens)
                sent_input_id_list.append(sub_input_ids)
                assert len(sub_tokens) == len(sub_input_ids)
                sent_id = sent_id + 1
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_token_list.append(sent_tokens_list)
            ctx_input_id_list.append(sent_input_id_list)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_with_answer = False
            answer_positions = []  ## answer position
            if norm_answer.strip() in ['yes', 'no', 'noanswer'] or (not answer_found_flag):
                ans_sub_tokens = None
                ans_input_ids = None
            else:
                ans_sub_tokens = tokenizer.tokenize(norm_answer, add_prefix_space=True)
                ans_input_ids = tokenizer.convert_tokens_to_ids(ans_sub_tokens)
                for sup_sent_idx, supp_sent_flag in answer_sent_flags:
                    supp_sent_encode_ids = sent_input_id_list[sup_sent_idx]
                    if supp_sent_flag:
                        answer_start_idx = sub_list_match_idx(target=ans_input_ids, source=supp_sent_encode_ids)
                        if answer_start_idx < 0:
                            ans_sub_tokens = tokenizer.tokenize(norm_answer.strip(), add_prefix_space=True)
                            ans_input_ids = tokenizer.convert_tokens_to_ids(ans_sub_tokens)
                            answer_start_idx = sub_list_match_idx(target=ans_input_ids, source=supp_sent_encode_ids)
                        answer_len = len(ans_input_ids)
                        assert answer_start_idx >= 0, "supp sent={} \n answer={} \n answer={} \n {} \n {}".format(
                            tokenizer.tokenizer.decode(supp_sent_encode_ids),
                            tokenizer.tokenizer.decode(ans_input_ids), norm_answer, supp_sent_encode_ids,
                            ans_sub_tokens)
                        ctx_with_answer = True
                        answer_positions.append((para_idx, sup_sent_idx, answer_start_idx, answer_start_idx + answer_len - 1))
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            assert len(para_names) == para_num
            assert len(sent_names) == sent_num
            assert len(ctx_token_list) == para_num and len(ctx_input_id_list) == para_num
            example = Example(qas_id=key,
                              qas_type=qas_type,
                              ctx_text=selected_contexts,
                              ctx_tokens=ctx_token_list,
                              ctx_input_ids=ctx_input_id_list,
                              para_names=para_names,
                              sup_para_id=sup_para_id,
                              sent_names=sent_names,
                              para_num=para_num,
                              sent_num=sent_num,
                              sup_fact_id=sup_facts_sent_id,
                              question_text=norm_question,
                              question_tokens=query_tokens,
                              question_input_ids=query_input_ids,
                              answer_text=norm_answer,
                              answer_tokens=ans_sub_tokens,
                              answer_input_ids=ans_input_ids,
                              answer_positions=answer_positions,
                              ctx_with_answer=ctx_with_answer)
            examples.append(example)
    return examples