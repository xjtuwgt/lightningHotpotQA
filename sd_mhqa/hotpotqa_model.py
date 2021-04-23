from torch import nn
from torch.autograd import Variable
import torch
from models.layers import OutputLayer
from torch import Tensor
import numpy as np
from sd_mhqa.transformer import EncoderLayer as Transformer_layer
from sd_mhqa.hotpotqa_evalutils import compute_loss

def para_sent_state_feature_extractor(batch, input_state: Tensor):
    sent_start, sent_end = batch['sent_start'], batch['sent_end'] - 1
    para_start, para_end = batch['para_start'], batch['para_end'] - 1

    batch_size, para_num, sent_num = para_start.shape[0], para_start.shape[1], sent_start.shape[1]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sent_batch_idx = torch.arange(0, batch_size, device=input_state.device).view(batch_size, 1).repeat(1, sent_num)
    sent_start_output = input_state[sent_batch_idx, sent_start]
    sent_end_output = input_state[sent_batch_idx, sent_end]
    sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    para_batch_idx = torch.arange(0, batch_size, device=input_state.device).view(batch_size, 1).repeat(1, para_num)
    para_start_output = input_state[para_batch_idx, para_start]
    para_end_output = input_state[para_batch_idx, para_end]
    para_state = torch.cat([para_start_output, para_end_output], dim=-1)

    state_dict = {'para_state': para_state, 'sent_state': sent_state}
    return state_dict

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
        return (para_prediction, sent_prediction)

class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    for answer span prediction
    """
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.config = config
        # self.hidden = config.hidden_dim
        self.hidden = config.transformer_hidden_dim

        self.start_linear = OutputLayer(self.hidden, config, num_answer=1)
        self.end_linear = OutputLayer(self.hidden, config, num_answer=1)
        self.type_linear = OutputLayer(self.hidden, config, num_answer=3)

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
            return (start_prediction, end_prediction, type_prediction)

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return (start_prediction, end_prediction, type_prediction, yp1, yp2)

class SDModel(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(SDModel, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.transformer_hidden_dim
        self.head_num = config.transformer_head_num
        self.linear_map = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=False)
        self.transformer_encoder = Transformer_layer(d_model=self.hidden_dim, ffn_hidden=4*self.hidden_dim, n_head=self.head_num)

        self.para_sent_predict_layer = ParaSentPredictionLayer(self.config, hidden_dim=2 * self.hidden_dim)
        self.predict_layer = PredictionLayer(self.config)

    def forward(self, encoder, batch, return_yp=False, return_cls=False):
        ####++++++++++++++++++++++++++++++++++++++
        ###############################################################################################################
        inputs = {'input_ids': batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'token_type_ids': batch['segment_idxs'] if self.config.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids
        ####++++++++++++++++++++++++++++++++++++++
        outputs = encoder(**inputs)
        print(type(outputs))
        print(len(outputs))
        batch['context_encoding'] = outputs[0]

        ####++++++++++++++++++++++++++++++++++++++
        batch['context_mask'] = batch['context_mask'].float().to(self.config.device)
        context_encoding = batch['context_encoding']
        input_state = self.linear_map(context_encoding)
        batch_mask = batch['context_mask'].unsqueeze(1)
        input_state = self.transformer_encoder.forward(x=input_state, src_mask=batch_mask)
        ####++++++++++++++++++++++++++++++++++++++
        ####++++++++++++++++++++++++++++++++++++++
        para_sent_state_dict = para_sent_state_feature_extractor(batch=batch, input_state=input_state)
        para_predictions, sent_predictions = self.para_sent_predict_layer.forward(state_dict=para_sent_state_dict)
        query_mapping = batch['query_mapping']
        if self.training:
            predictions = self.predict_layer.forward(batch=batch, context_input=input_state,
                                                 packing_mask=query_mapping, return_yp=False)
        else:
            predictions = self.predict_layer.forward(batch=batch, context_input=input_state,
                                                     packing_mask=query_mapping, return_yp=True)
        if not self.training:
            start, end, q_type, yp1, yp2 = predictions
            if return_yp:
                return start, end, q_type, para_predictions, sent_predictions, yp1, yp2
            else:
                return start, end, q_type, para_predictions, sent_predictions
        else:
            start, end, q_type = predictions
            loss_list = compute_loss(self.config, batch, start, end, para_predictions, sent_predictions, q_type)
            return loss_list