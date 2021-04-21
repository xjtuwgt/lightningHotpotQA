from models.layers import BiAttention, LSTMWrapper
from torch import nn
from torch.autograd import Variable
import torch
from models.layers import OutputLayer
import numpy as np

class SentPredictionLayer(nn.Module):
    def __init__(self, config, hidden_dim):
        super(SentPredictionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)

    def forward(self, sent_state):
        sent_logit = self.sent_mlp(sent_state)
        sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
        sent_prediction = torch.cat([sent_logits_aux, sent_logit], dim=-1).contiguous()
        return sent_prediction

def init_sent_para_feature(batch, input_state, hidden_dim):
    sent_start_mapping = batch['sent_start_mapping']
    sent_end_mapping = batch['sent_end_mapping']
    sent_start_output = torch.bmm(sent_start_mapping, input_state[:, :, hidden_dim:])  # N x max_sent x d
    sent_end_output = torch.bmm(sent_end_mapping, input_state[:, :, :hidden_dim])  # N x max_sent x d
    sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)  # N x max_sent x 2d
    return sent_state

class ParaSentPredictionLayer(nn.Module):
    def __init__(self, config, hidden_dim):
        super(ParaSentPredictionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.para_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)

    def forward(self, state_dict):
        para_state = state_dict['para_state']
        sent_state = state_dict['sent_state']

        N, _, _ = para_state.size()
        sent_logit = self.sent_mlp(sent_state)
        para_logit = self.para_mlp(para_state)

        para_logits_aux = Variable(para_logit.data.new(para_logit.size(0), para_logit.size(1), 1).zero_())
        para_prediction = torch.cat([para_logits_aux, para_logit], dim=-1).contiguous()

        sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
        sent_prediction = torch.cat([sent_logits_aux, sent_logit], dim=-1).contiguous()
        return para_prediction, sent_prediction

class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    for answer span prediction
    """
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.config = config
        input_dim = config.ctx_attn_hidden_dim
        h_dim = config.hidden_dim

        self.hidden = h_dim

        self.start_linear = OutputLayer(input_dim, config, num_answer=1)
        self.end_linear = OutputLayer(input_dim, config, num_answer=1)
        self.type_linear = OutputLayer(input_dim, config, num_answer=4)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, packing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        type_prediction = self.type_linear(context_input[:, 0, :])

        if not return_yp:
            return start_prediction, end_prediction, type_prediction

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction, end_prediction, type_prediction, yp1, yp2

class HotPotQAModel(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HotPotQAModel, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length
        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.hidden_dim = config.hidden_dim
        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=config.lstm_layer,
                                     dropout=config.lstm_drop) ### output: 2 * self.hidden_dim

    def forward(self, batch, return_yp=False, return_cls=False):
        query_mapping = batch['query_mapping']
        context_encoding = batch['context_encoding']
        # print('context encode', context_encoding.shape)
        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)
        input_state = self.bi_attn_linear(attn_output) # N x L x d
        input_state = self.sent_lstm(input_state, batch['context_lens'])
        ###############################################################################################################
        return