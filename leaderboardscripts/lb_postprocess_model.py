import torch
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
import numpy as np
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


class OutputLayer(nn.Module):
    def __init__(self, hidden_dim, trans_drop=0.25, num_answer=1):
        super(OutputLayer, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            LayerNorm(hidden_dim*2, eps=1e-12),
            nn.Dropout(trans_drop),
            #+++++++++
            nn.Linear(2*hidden_dim, hidden_dim*2),
            nn.ReLU(),
            LayerNorm(hidden_dim * 2, eps=1e-12),
            nn.Dropout(trans_drop),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            LayerNorm(hidden_dim * 2, eps=1e-12),
            nn.Dropout(trans_drop),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),

            nn.ReLU(),
            LayerNorm(hidden_dim * 2, eps=1e-12),
            nn.Dropout(trans_drop),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),

            # nn.Linear(2 * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            LayerNorm(hidden_dim * 2, eps=1e-12),
            nn.Dropout(trans_drop),
            nn.Linear(hidden_dim * 2, num_answer),
            #+++++++++
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)

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
        self.score_map = PositionwiseFeedForward(model_dim=self.score_dim,
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

def loss_computation(scores, y_min, y_max):
    # p_score = F.sigmoid(scores)
    p_score = scores.squeeze(-1)
    # print(y_min)
    # print(y_max)
    # print(p_score)
    loss = F.relu(p_score - y_max) + F.relu(y_min - p_score)
    # loss = F.relu(torch.tanh(p_score) - torch.tanh(y_max)) + F.relu(torch.tanh(y_min) - torch.tanh(p_score))
    loss = loss * loss
    # loss = loss.mean()
    loss = loss.sum()
    return loss
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# def ce_loss_computation(scores, y_min, y_max, score_gold):
#     criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
#     score_aux = Variable(scores.data.new(scores.size(0), scores.size(1)).zero_())
#     score_pred = torch.cat([score_aux, scores], dim=-1).contiguous()
#     loss_sup = criterion(score_pred, score_gold.long())
#     p_score = torch.sigmoid(scores.squeeze(-1))
#     loss_range = F.relu(p_score - torch.sigmoid(y_max)) + F.relu(torch.sigmoid(y_min) - p_score)
#     loss_range = loss_range.mean()
#     loss = 0*loss_sup + loss_range
#     return loss, loss_sup, loss_range


class RangeSeqModel(nn.Module):
    def __init__(self, args):
        super(RangeSeqModel, self).__init__()
        self.args = args
        self.cls_emb_dim = self.args.cls_emb_dim
        self.emb_dim = self.args.emb_dim
        self.score_dim = self.emb_dim - self.cls_emb_dim
        self.hid_dim = self.args.hid_dim

        self.cls_map = PositionwiseFeedForward(model_dim=self.cls_emb_dim,
                                               d_hidden=1024, out_dim=self.hid_dim)
        self.score_map = PositionwiseFeedForward(model_dim=self.score_dim,
                                               d_hidden=1024, out_dim=self.hid_dim)

        self.start_linear = OutputLayer(self.hidden, trans_drop=self.args.feat_drop, num_answer=self.args.interval_number)
        self.end_linear = OutputLayer(self.hidden, trans_drop=self.args.feat_drop, num_answer=self.args.interval_number)

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
        cls_map_emb = self.cls_map.forward(cls_x)
        score_map_emb = self.score_map.forward(score_x)
        x_emb = torch.cat([cls_map_emb, score_map_emb], dim=-1)
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


class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    for answer span prediction
    """
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.config = config
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
        # print('outer', outer.shape)
        outer_mask = self.get_output_mask(outer)
        # print('outer mask', outer_mask.shape)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        return (start_prediction, end_prediction, type_prediction, yp1, yp2)

def seq_loss_computation(start, end, batch):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    loss_span = (criterion(start, batch['y1']) + criterion(end, batch['y2']))
    return loss_span