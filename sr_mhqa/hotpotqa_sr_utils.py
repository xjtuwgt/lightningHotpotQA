from sd_mhqa.hotpotqaUtils import json_loader
from sd_mhqa.hotpotqaUtils import tokenize_text
from sd_mhqa.hotpotqaUtils import ranked_context_processing
from sd_mhqa.hotpotqaUtils import sub_list_match_idx
from sd_mhqa.hotpotqaUtils import normalize_text
from sr_mhqa.hotpotqa_sr_data_structure import Example
from numpy import random
from tqdm import tqdm

def hotpot_answer_neg_sents_tokenizer(split_para_file: str,
                            full_file: str,
                            tokenizer,
                            cls_token='[CLS]',
                            sep_token='[SEP]',
                            is_roberta=False,
                            data_source_type=None):
    split_para_rank_data = json_loader(json_file_name=split_para_file)
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
        split_paras = split_para_rank_data[key]
        assert len(split_paras) == 3
        selected_para_titles = [_[0] for _ in split_paras[0]]
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
                        # answer_positions.append((para_idx, sup_sent_idx, answer_start_idx, answer_start_idx + answer_len))
                        answer_positions.append(
                            (title, sup_sent_idx, answer_start_idx, answer_start_idx + answer_len))
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            assert len(para_names) == para_num
            assert len(sent_names) == sent_num
            assert len(ctx_token_list) == para_num and len(ctx_input_id_list) == para_num
        ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++diff the rankers
        if data_source_type is not None:
            key = key + "_" + data_source_type
        ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        negative_selected_titles = [_[0] for _ in split_paras[2]]
        neg_ctx_text, neg_ctx_tokens, neg_ctx_input_ids = negative_context_processing(row=row, tokenizer=tokenizer,
                                                                                      is_roberta=is_roberta,
                                                                                      sep_token=sep_token, selected_para_titles=negative_selected_titles)
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
                          ctx_with_answer=ctx_with_answer,
                          neg_ctx_text=neg_ctx_text,
                          neg_ctx_tokens=neg_ctx_tokens,
                          neg_ctx_input_ids=neg_ctx_input_ids)
        examples.append(example)
    print('Answer not found = {}'.format(answer_not_found_count))
    return examples

def negative_context_processing(row, tokenizer, selected_para_titles, sep_token, is_roberta):
    ################################################################################################################
    selected_contexts = []
    context_dict = dict(row['context'])
    for title in selected_para_titles:
        para_text = context_dict[title]
        para_text_lower = [normalize_text(text=sent) for sent in para_text]
        selected_contexts.append([title, para_text_lower, 0, [], False]) ## no support para
    ctx_token_list = []
    ctx_input_id_list = []
    for para_idx, para_tuple in enumerate(selected_contexts):
        title, sents, _, _, _ = para_tuple
        sent_tokens_list = []
        sent_input_id_list = []
        for local_sent_id, sent_text in enumerate(sents):
            sent_words, sent_sub_tokens = tokenize_text(text=sent_text, tokenizer=tokenizer, is_roberta=is_roberta)
            if is_roberta:
                sent_sub_tokens.append(sep_token)
            sub_input_ids = tokenizer.convert_tokens_to_ids(sent_sub_tokens)
            assert len(sub_input_ids) == len(sent_sub_tokens)
            sent_tokens_list.append(sent_sub_tokens)
            sent_input_id_list.append(sub_input_ids)
            assert len(sent_sub_tokens) == len(sub_input_ids)
        ctx_token_list.append(sent_tokens_list)
        ctx_input_id_list.append(sent_input_id_list)
    assert len(ctx_token_list) == len(ctx_input_id_list) and len(ctx_token_list) == len(selected_contexts)
    return selected_contexts, ctx_token_list, ctx_input_id_list

################################################################################################################
def neg_sentence_sampler(neg_ctx_tokens: list, neg_ctx_input_ids: list):
    assert len(neg_ctx_tokens) == len(neg_ctx_input_ids)
    para_num = len(neg_ctx_tokens)
    rand_para_idx = random.randint(0, para_num)
    neg_ctx_sents = neg_ctx_tokens[rand_para_idx]
    neg_ctx_sent_input_ids = neg_ctx_input_ids[rand_para_idx]
    assert len(neg_ctx_sents) == len(neg_ctx_sent_input_ids)
    sent_num = len(neg_ctx_sents)
    rand_sent_idx = random.randint(0, sent_num)
    neg_sent_tokens = neg_ctx_sents[rand_sent_idx]
    neg_sent_input_ids = neg_ctx_sent_input_ids[rand_sent_idx]
    assert len(neg_sent_tokens) == len(neg_sent_input_ids)
    return neg_sent_tokens, neg_sent_input_ids
################################################################################################################
def example_sent_replacement(case: Example, replace_ratio:float = 0.1):
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
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    neg_ctx_text = case.neg_ctx_text
    neg_ctx_tokens = case.neg_ctx_tokens
    neg_ctx_input_ids = case.neg_ctx_input_ids
    ###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    assert para_num == len(ctx_tokens) and para_num == len(ctx_input_ids) and para_num == len(para_names)
    assert sent_num == len(sent_names)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sent_name_to_id_dict = dict([((x[1][0], x[1][1]), x[0]) for x in enumerate(sent_names)])
    replace_sent_idxs = []
    replace_ctx_tokens = []
    replace_ctx_input_ids = []
    for para_idx, para_name in enumerate(para_names):
        replace_para_ctx_tokens, replace_para_input_ids = [], []
        for sent_idx, (sent_sub_token, sent_inp_ids) in enumerate(zip(ctx_tokens[para_idx], ctx_input_ids[para_idx])):
            abs_sent_idx = sent_name_to_id_dict[(para_name, sent_idx)]
            if abs_sent_idx not in sup_fact_id:
                rand_s_i = random.rand()
                if rand_s_i > replace_ratio:
                    replace_para_ctx_tokens.append(sent_sub_token)
                    replace_para_input_ids.append(sent_inp_ids)
                    replace_sent_idxs.append(0)
                else:
                    neg_sent_tokens, neg_sent_input_ids = neg_sentence_sampler(neg_ctx_tokens=neg_ctx_tokens,
                                                                               neg_ctx_input_ids=neg_ctx_input_ids)
                    replace_para_ctx_tokens.append(neg_sent_tokens)
                    replace_para_input_ids.append(neg_sent_input_ids)
                    replace_sent_idxs.append(1)
            else:
                replace_para_ctx_tokens.append(sent_sub_token)
                replace_para_input_ids.append(sent_inp_ids)
                replace_sent_idxs.append(0)
        replace_ctx_tokens.append(replace_para_ctx_tokens)
        replace_ctx_input_ids.append(replace_para_input_ids)
    assert len(replace_ctx_tokens) == len(replace_ctx_input_ids)
    assert len(sent_names) == len(replace_sent_idxs)
    replace_example = Example(
        qas_id=qas_id,
        qas_type=qas_type,
        ctx_text=ctx_text,
        ctx_tokens=replace_ctx_tokens,
        ctx_input_ids=replace_ctx_input_ids,
        para_names=para_names,
        sent_names=sent_names,
        sup_para_id=sup_para_id,
        para_num=para_num,
        sent_num=sent_num,
        sup_fact_id=sup_fact_id,
        question_text=question_text,
        question_tokens=question_tokens,
        question_input_ids=question_input_ids,
        answer_text=answer_text,
        answer_tokens=answer_tokens,
        answer_input_ids=answer_input_ids,
        answer_positions=answer_positions,
        ctx_with_answer=ctx_with_answer,
        neg_ctx_text=neg_ctx_text,
        neg_ctx_tokens=neg_ctx_tokens,
        neg_ctx_input_ids=neg_ctx_input_ids)
    return replace_example, replace_sent_idxs
###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++