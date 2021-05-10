import torch
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
import numpy as np
from leaderboardscripts.lb_postprocess_model_utils import AddNorm, PoswiseFeedForwardNet
from torch.autograd import Variable
IGNORE_INDEX = -100

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, out_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(model_dim, d_hidden * 4),
            nn.ReLU(),
            LayerNorm(d_hidden * 4, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 4, out_dim),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class RangeModel(nn.Module):
    def __init__(self, args):
        super(RangeModel, self).__init__()
        self.args = args
        self.cls_emb_dim = self.args.cls_emb_dim
        self.emb_dim = self.args.emb_dim
        self.score_dim = self.emb_dim - self.cls_emb_dim
        self.hid_dim = self.args.hid_dim

        self.cls_map = PositionwiseFeedForward(model_dim=self.cls_emb_dim,
                                               d_hidden=1024, out_dim=self.hid_dim)
        self.score_map = PositionwiseFeedForward(model_dim= self.score_dim,
                                               d_hidden=1024, out_dim=self.hid_dim)
        self.threshold_score_func = OutputLayer(hidden_dim=2 * self.hid_dim,
                                           trans_drop=self.args.feat_drop,
                                           num_answer=1)
    def forward(self, x: T):
        assert x.shape[1] == self.emb_dim
        cls_x = x[:,:self.cls_emb_dim]
        score_x = x[:,self.cls_emb_dim:]
        cls_map_emb = self.cls_map.forward(cls_x)
        score_map_emb = self.score_map.forward(score_x)
        x_emb = torch.cat([cls_map_emb, score_map_emb], dim=-1)
        scores = self.threshold_score_func.forward(x_emb)
        return scores

def loss_computation(scores, y_min, y_max, weight=None):
    p_score = scores.squeeze(-1)
    if weight is None:
        loss = F.relu(p_score - y_max) + F.relu(y_min - p_score)
    else:
        loss = F.relu(p_score - y_max) + F.relu(y_min - p_score)
        loss = loss * weight
    # loss = F.relu(torch.tanh(p_score) - torch.tanh(y_max)) + F.relu(torch.tanh(y_min) - torch.tanh(p_score))
    # loss = loss * loss
    loss = loss.mean()
    # loss = loss.sum()
    return loss

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super(TransformerEncoder, self).__init__()



class ConvEncoder(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p, layer_num):
        super(ConvEncoder, self).__init__()
        self.conv_encoder = nn.ModuleList()
        for i in range(layer_num):
            self.conv_encoder.append(AddNorm(PoswiseFeedForwardNet(d_model, d_ff, dropout_p, 'conv'), d_model))
    def forward(self, inputs: T) -> T:
        """
        :param inputs: batch size x 2 x dim
        :return:
        """
        output = inputs
        for layer in self.conv_encoder:
            output = layer(output)
        return output.flatten(start_dim=1)

class MLPEncoder(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p, layer_num):
        super(MLPEncoder, self).__init__()
        self.mlp_encoder = nn.ModuleList()
        for i in range(layer_num):
            self.mlp_encoder.append(AddNorm(PoswiseFeedForwardNet(d_model, d_ff, dropout_p, 'ff'), d_model))
    def forward(self, inputs: T) -> T:
        """
        :param inputs: batch_size x (2 * dim)
        :return:
        """
        output = inputs
        for layer in self.conv_encoder:
            output = layer(output)
        return output


class OutputLayer(nn.Module):
    def __init__(self, hidden_dim, trans_drop=0.35, num_answer=1):
        super(OutputLayer, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            LayerNorm(hidden_dim*2, eps=1e-12),
            nn.Dropout(trans_drop),
            nn.Linear(hidden_dim * 2, num_answer)
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class RangeSeqModel(nn.Module):
    def __init__(self, args):
        super(RangeSeqModel, self).__init__()
        self.args = args
        self.cls_emb_dim = self.args.cls_emb_dim
        self.emb_dim = self.args.emb_dim
        self.score_dim = self.emb_dim - self.cls_emb_dim
        self.hid_dim = self.args.hid_dim

        self.cls_map = PositionwiseFeedForward(model_dim=self.cls_emb_dim,
                                               d_hidden=2048, out_dim=self.hid_dim)
        self.score_map = PositionwiseFeedForward(model_dim= 3 * self.score_dim,
                                               d_hidden=2048, out_dim=self.hid_dim)
        ##+++++++++++++++++++++++++++++++++++++++++
        self.encoder_type = self.args.encoder_type
        if self.encoder_type == 'ff':
            self.encoder = MLPEncoder(d_model= 2*self.hid_dim, d_ff=2048, dropout_p=self.args.encoder_drop_out,
                                      layer_num=self.args.encoder_layer)
        elif self.encoder_type == 'conv':
            self.encoder = ConvEncoder(d_model= self.hid_dim, d_ff=2048, dropout_p=self.args.encoder_drop_out,
                                      layer_num=self.args.encoder_layer)
        elif self.encoder_type == 'transformer':
            self.encoder = None
        else:
            raise '{} encoder is not supported'.format(self.encoder_type)
        ##+++++++++++++++++++++++++++++++++++++++++

        self.start_linear = OutputLayer(2 * self.hid_dim, trans_drop=self.args.feat_drop, num_answer=self.args.interval_number)
        self.end_linear = OutputLayer(2 * self.hid_dim, trans_drop=self.args.feat_drop, num_answer=self.args.interval_number)

        self.cache_S = 0
        self.cache_mask = None
        self.span_window_size = self.args.span_window_size

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), self.span_window_size)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, x: T, return_yp=False):
        assert x.shape[1] == self.emb_dim
        cls_x = x[:,:self.cls_emb_dim]
        score_x = x[:,self.cls_emb_dim:]
        tanh_score_x = F.tanh(score_x)
        power_score = torch.pow(score_x, 2)
        score_x = torch.cat([score_x, tanh_score_x, power_score], dim=-1)
        cls_map_emb = self.cls_map.forward(cls_x)
        score_map_emb = self.score_map.forward(score_x)
        ##+++++++++++++++++++++++++++++++++++++++++
        if self.encoder_type == 'ff':
            x_emb = torch.cat([cls_map_emb, score_map_emb], dim=-1)
            x_emb = self.encoder.forward(x_emb)
        elif self.encoder_type == 'conv':
            x_emb = torch.stack([cls_map_emb, score_map_emb], dim=1)
            x_emb = self.encoder.forward(x_emb)
        elif self.encoder_type == 'transformer':
            x_emb = torch.cat([cls_map_emb, score_map_emb], dim=-1)
        else:
            raise '{} encoder is not supported'.format(self.encoder_type)
        ##+++++++++++++++++++++++++++++++++++++++++

        start_prediction_scores = self.start_linear(x_emb)
        end_prediction_scores = self.end_linear(x_emb)

        if not return_yp:
            return (start_prediction_scores, end_prediction_scores)

        outer = start_prediction_scores[:, :, None] + end_prediction_scores[:, None]
        outer_mask = self.get_output_mask(outer)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return start_prediction_scores, end_prediction_scores, yp1, yp2

def seq_loss_computation(start, end, batch, weight=False):
    if not weight:
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
        loss_span = criterion(start, batch['y_1']) + criterion(end, batch['y_2'])
    else:
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        loss_span = (criterion(start, batch['y_1']) + criterion(end, batch['y_2'])) * batch['weight']
        loss_span = torch.mean(loss_span)
    return loss_span