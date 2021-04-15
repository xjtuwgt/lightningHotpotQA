class Example(object):
    def __init__(self,
                 qas_id,
                 qas_type,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 para_names,
                 sup_fact_id,
                 sup_para_id,
                 para_start_end_position,
                 sent_start_end_position,
                 question_text,
                 ctx_text,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.para_names = para_names
        self.sup_fact_id = sup_fact_id
        self.sup_para_id = sup_para_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.ctx_text = ctx_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
