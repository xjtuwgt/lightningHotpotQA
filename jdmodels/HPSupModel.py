from models.layers import BiAttention, LSTMWrapper
from torch import nn
from torch.autograd import Variable
import torch
from models.layers import OutputLayer

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

def init_sent_feature(batch, input_state, hidden_dim):
    sent_start_mapping = batch['sent_start_mapping']
    sent_end_mapping = batch['sent_end_mapping']
    sent_start_output = torch.bmm(sent_start_mapping, input_state[:, :, hidden_dim:])  # N x max_sent x d
    sent_end_output = torch.bmm(sent_end_mapping, input_state[:, :, :hidden_dim])  # N x max_sent x d
    sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)  # N x max_sent x 2d
    return sent_state

class HierarchicalGraphNetwork(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
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
        self.sent_predict_layer = SentPredictionLayer(self.config, hidden_dim=self.hidden_dim)

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
        ################################################################################################################
        sent_state = init_sent_feature(batch=batch, input_state=input_state, hidden_dim=self.hidden_dim)
        sent_predictions = self.sent_predict_layer.forward(sent_state=sent_state)
        return sent_predictions