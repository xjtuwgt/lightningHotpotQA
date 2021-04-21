import json
import pandas as pd
from pandas import DataFrame
from time import time
from tqdm import tqdm
import itertools
from HotpotQAModel.hotpotqa_data_structure import Example
import spacy
import re
nlp = spacy.load("en_core_web_lg", disable=['tagger', 'parser'])
infix_re = re.compile(r'''[-—–~]''')
nlp.tokenizer.infix_finditer = infix_re.finditer
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def json_loader(json_file_name: str):
    with open(json_file_name, 'r', encoding='utf-8') as reader:
        json_data = json.load(reader)
    return json_data
def loadWikiData(json_file_name: str)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(json_file_name, orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame
def split_sent(sent: str):
    nlp_doc = nlp(sent)
    words = []
    for token in nlp_doc:
        words.append(token.text)
    return words
def tokenize_text(text: str, tokenizer, is_roberta):
    words = split_sent(sent=text)
    sub_tokens = []
    for word in words:
        if is_roberta:
            sub_toks = tokenizer.tokenize(word, add_prefix_space=True)
        else:
            sub_toks = tokenizer.tokenize(word)
        sub_tokens += sub_toks
    return words, sub_tokens
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
def find_answer_span(norm_answer, sentence, tokenizer, is_roberta):
    _, ans_sub_tokens = tokenize_text(text=norm_answer, tokenizer=tokenizer, is_roberta=is_roberta)
    _, sent_sub_tokens = tokenize_text(text=sentence, tokenizer=tokenizer, is_roberta=is_roberta)
    idx = sub_list_match_idx(target=ans_sub_tokens, source=sent_sub_tokens)
    flag = idx >= 0
    return flag, ans_sub_tokens, sent_sub_tokens
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
def ranked_context_processing(row, tokenizer, selected_para_titles, is_roberta):
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
                            encode_has_answer, X, Y = find_answer_span(norm_answer.strip(), supp_sent, tokenizer, is_roberta)
                            if not encode_has_answer:
                                encode_has_answer, X, Y = find_answer_span(norm_answer, supp_sent, tokenizer, is_roberta)
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
    yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
    if not answer_found_flag and (norm_answer.strip() not in ['yes', 'no']):
        norm_answer = 'noanswer'
    return norm_question, norm_answer, selected_contexts, supporting_facts_filtered, yes_no_flag, answer_found_flag
#=======================================================================================================================
def hotpot_answer_tokenizer(para_file: str,
                            full_file: str,
                            tokenizer,
                            cls_token='[CLS]',
                            sep_token='[SEP]',
                            is_roberta=False,
                            data_source_type=None):
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
            ranked_context_processing(row=row, tokenizer=tokenizer, selected_para_titles=selected_para_titles, is_roberta=is_roberta)
        # print(yes_no_flag, answer_found_flag)
        if not answer_found_flag and not yes_no_flag:
            answer_not_found_count = answer_not_found_count + 1
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_tokens = [cls_token]
        query_words, query_sub_tokens = tokenize_text(text=norm_question, tokenizer=tokenizer, is_roberta=is_roberta)
        query_tokens += query_sub_tokens
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
        ctx_with_answer = False
        answer_positions = []  ## answer position
        ans_sub_tokens = []
        ans_input_ids = []
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
                sent_words, sent_sub_tokens = tokenize_text(text=sent_text, tokenizer=tokenizer, is_roberta=is_roberta)
                if is_roberta:
                    sent_sub_tokens.append(sep_token)
                sub_input_ids = tokenizer.convert_tokens_to_ids(sent_sub_tokens)
                assert len(sub_input_ids) == len(sent_sub_tokens)
                sent_tokens_list.append(sent_sub_tokens)
                sent_input_id_list.append(sub_input_ids)
                assert len(sent_sub_tokens) == len(sub_input_ids)
                sent_id = sent_id + 1
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_token_list.append(sent_tokens_list)
            ctx_input_id_list.append(sent_input_id_list)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if (norm_answer.strip() not in ['yes', 'no', 'noanswer']) and answer_found_flag:
                ans_words, ans_sub_tokens = tokenize_text(text=norm_answer, tokenizer=tokenizer, is_roberta=is_roberta)
                ans_input_ids = tokenizer.convert_tokens_to_ids(ans_sub_tokens)
                for sup_sent_idx, supp_sent_flag in answer_sent_flags:
                    supp_sent_encode_ids = sent_input_id_list[sup_sent_idx]
                    if supp_sent_flag:
                        answer_start_idx = sub_list_match_idx(target=ans_input_ids, source=supp_sent_encode_ids)
                        if answer_start_idx < 0:
                            ans_words, ans_sub_tokens = tokenize_text(text=norm_answer.strip(), tokenizer=tokenizer,
                                                                      is_roberta=is_roberta)
                            ans_input_ids = tokenizer.convert_tokens_to_ids(ans_sub_tokens)
                            answer_start_idx = sub_list_match_idx(target=ans_input_ids, source=supp_sent_encode_ids)
                        answer_len = len(ans_input_ids)
                        assert answer_start_idx >= 0, "supp sent={} \n answer={} \n answer={} \n {} \n {}".format(tokenizer.decode(supp_sent_encode_ids),
                            tokenizer.decode(ans_input_ids), norm_answer, supp_sent_encode_ids, ans_sub_tokens)
                        ctx_with_answer = True
                        answer_positions.append((para_idx, sup_sent_idx, answer_start_idx, answer_start_idx + answer_len))
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            assert len(para_names) == para_num
            assert len(sent_names) == sent_num
            assert len(ctx_token_list) == para_num and len(ctx_input_id_list) == para_num
        ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++diff the rankers
        if data_source_type is not None:
            key = key + "_" + data_source_type
        ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    print('Answer not found = {}'.format(answer_not_found_count))
    return examples