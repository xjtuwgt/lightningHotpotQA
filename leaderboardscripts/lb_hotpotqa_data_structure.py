
def get_cached_filename(f_type, config):
    f_type_set = {'examples', 'features', 'graphs',

                  'hgn_examples', 'hgn_features', 'hgn_graphs',
                  'hgn_reverse_examples', 'hgn_reverse_features', 'hgn_reverse_graphs',

                  'hgn_low_examples', 'hgn_low_features', 'hgn_low_graphs',
                  'hgn_low_reverse_examples', 'hgn_low_reverse_features', 'hgn_low_reverse_graphs',

                  'hgn_low_sae_examples', 'hgn_low_sae_features', 'hgn_low_sae_graphs',
                  'hgn_low_sae_reverse_examples', 'hgn_low_sae_reverse_features', 'hgn_low_sae_reverse_graphs',

                  'long_examples', 'long_features', 'long_graphs',
                  'long_reverse_examples', 'long_reverse_features', 'long_reverse_graphs',

                  'long_low_examples', 'long_low_features', 'long_low_graphs',
                  'long_low_reverse_examples', 'long_low_reverse_features', 'long_low_reverse_graphs',

                  'long_low_sae_examples', 'long_low_sae_features', 'long_low_sae_graphs',
                  'long_low_sae_reverse_examples', 'long_low_sae_reverse_features', 'long_low_sae_reverse_graphs',

                  'docred_low_examples', 'docred_low_features', 'docred_low_graphs',
                  'docred_low_sae_examples', 'docred_low_sae_features', 'docred_low_sae_graphs',

                  'hgn_docred_low_examples', 'hgn_docred_low_features', 'hgn_docred_low_graphs',
                  'hgn_docred_low_sae_examples', 'hgn_docred_low_sae_features', 'hgn_docred_low_sae_graphs',

                  'long_docred_low_examples', 'long_docred_low_features', 'long_docred_low_graphs',
                  'long_docred_low_sae_examples', 'long_docred_low_sae_features', 'long_docred_low_sae_graphs',

                  'hgn_long_docred_low_examples', 'hgn_long_docred_low_features', 'hgn_long_docred_low_graphs',
                  'hgn_long_docred_low_sae_examples', 'hgn_long_docred_low_sae_features', 'hgn_long_docred_low_sae_graphs',

                  'hgn_long_low_examples', 'hgn_long_low_features', 'hgn_long_low_graphs',
                  'hgn_long_low_sae_examples', 'hgn_long_low_sae_features', 'hgn_long_low_sae_graphs',

                  'oracle_features', 'oracle_graphs', 'oracle_examples',
                  'oracle_sae_features', 'oracle_sae_graphs', 'oracle_sae_examples',

                  'oracle_features', 'oracle_graphs', 'oracle_examples',
                  'oracle_sae_features', 'oracle_sae_graphs', 'oracle_sae_examples',

                  'hgn_low_reranker2_examples', 'hgn_low_reranker2_features', 'hgn_low_reranker2_graphs',
                  'hgn_low_reranker3_examples', 'hgn_low_reranker3_features', 'hgn_low_reranker3_graphs',

                  'hgn_low_sae_reranker2_examples', 'hgn_low_sae_reranker2_features', 'hgn_low_sae_reranker2_graphs',
                  'hgn_low_sae_reranker3_examples', 'hgn_low_sae_reranker3_features', 'hgn_low_sae_reranker3_graphs',

                  'long_low_reranker2_examples', 'long_low_reranker2_features', 'long_low_reranker2_graphs',
                  'long_low_reranker3_examples', 'long_low_reranker3_features', 'long_low_reranker3_graphs',

                  'long_low_sae_reranker2_examples', 'long_low_sae_reranker2_features', 'long_low_sae_reranker2_graphs',
                  'long_low_sae_reranker3_examples', 'long_low_sae_reranker3_features', 'long_low_sae_reranker3_graphs',
                  #++++

                  'roberta_hgn_low_reranker2_examples', 'roberta_hgn_low_reranker2_features', 'roberta_hgn_low_reranker2_graphs',
                  'roberta_hgn_low_reranker3_examples', 'roberta_hgn_low_reranker3_features', 'roberta_hgn_low_reranker3_graphs',

                  'roberta_hgn_low_sae_reranker2_examples', 'roberta_hgn_low_sae_reranker2_features', 'roberta_hgn_low_sae_reranker2_graphs',
                  'roberta_hgn_low_sae_reranker3_examples', 'roberta_hgn_low_sae_reranker3_features', 'roberta_hgn_low_sae_reranker3_graphs',

                  'albert_long_low_reranker2_examples', 'albert_long_low_reranker2_features', 'albert_long_low_reranker2_graphs',
                  'albert_long_low_reranker3_examples', 'albert_long_low_reranker3_features', 'albert_long_low_reranker3_graphs',

                  'albert_long_low_sae_reranker2_examples', 'albert_long_low_sae_reranker2_features', 'albert_long_low_sae_reranker2_graphs',
                  'albert_long_low_sae_reranker3_examples', 'albert_long_low_sae_reranker3_features', 'albert_long_low_sae_reranker3_graphs',

                  } #### ranker: hgn, longformer; case: lowercase, cased; graph: whether sae-graph
    assert f_type in f_type_set
    return f"cached_{f_type}_{config.model_type}_{config.max_seq_length}_{config.max_query_length}.pkl.gz"

class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 question_tokens,
                 doc_tokens,
                 sent_num,
                 sent_names,
                 para_names,
                 ques_entities_text,
                 ctx_entities_text,
                 para_start_end_position,
                 sent_start_end_position,
                 ques_entity_start_end_position,
                 ctx_entity_start_end_position,
                 question_text,
                 question_word_to_char_idx,
                 ctx_text,
                 ctx_word_to_char_idx,
                 edges=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.para_names = para_names
        self.ques_entities_text = ques_entities_text
        self.ctx_entities_text = ctx_entities_text
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.ques_entity_start_end_position = ques_entity_start_end_position
        self.ctx_entity_start_end_position = ctx_entity_start_end_position
        self.question_word_to_char_idx = question_word_to_char_idx
        self.ctx_text = ctx_text
        self.ctx_word_to_char_idx = ctx_word_to_char_idx
        self.edges = edges

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 para_spans,
                 sent_spans,
                 entity_spans,
                 q_entity_cnt,
                 token_to_orig_map,
                 edges=None):
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.para_spans = para_spans
        self.sent_spans = sent_spans
        self.entity_spans = entity_spans
        self.q_entity_cnt = q_entity_cnt

        self.edges = edges
        self.token_to_orig_map = token_to_orig_map