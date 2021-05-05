import torch
from torch import Tensor as T
from torch import nn
import torch.nn.functional as F
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

# class PositionwiseFeedForward(nn.Module):
#     "Implements FFN equation."
#     def __init__(self, model_dim, d_hidden, out_dim, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(model_dim, d_hidden)
#         self.w_2 = nn.Linear(d_hidden, out_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.init()
#
#     def forward(self, x):
#         return self.w_2(self.dropout(F.relu(self.w_1(x))))
#
#     def init(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.w_1.weight, gain=gain)
#         nn.init.xavier_normal_(self.w_2.weight, gain=gain)

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

        # self.threshold_score_func = OutputLayer(hidden_dim=self.score_dim,
        #                                         trans_drop=self.args.feat_drop,
        #                                         num_answer=1)

        # self.threshold_score_func = OutputLayer(hidden_dim=self.cls_emb_dim,
        #                                         trans_drop=self.args.feat_drop,
        #                                         num_answer=1)
        # self.threshold_score_func = OutputLayer(hidden_dim=self.hid_dim,
        #                                         trans_drop=self.args.feat_drop,
        #                                         num_answer=1)

        # self.threshold_score_func = OutputLayer(hidden_dim=self.emb_dim,
        #                                         trans_drop=self.args.feat_drop,
        #                                         num_answer=1)
    def forward(self, x: T):
        assert x.shape[1] == self.emb_dim
        cls_x = x[:,:self.cls_emb_dim]
        score_x = x[:,self.cls_emb_dim:]
        cls_map_emb = self.cls_map.forward(cls_x)
        score_map_emb = self.score_map.forward(score_x)
        x_emb = torch.cat([cls_map_emb, score_map_emb], dim=-1)
        # scores = self.threshold_score_func.forward(score_x)
        # scores = self.threshold_score_func.forward(x)
        # scores = self.threshold_score_func.forward(cls_x)
        # scores = self.threshold_score_func.forward(score_map_emb)
        # scores = self.threshold_score_func.forward(cls_map_emb)
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

def ce_loss_computation(scores, y_min, y_max, score_gold):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    score_aux = Variable(scores.data.new(scores.size(0), scores.size(1)).zero_())
    score_pred = torch.cat([score_aux, scores], dim=-1).contiguous()
    loss_sup = criterion(score_pred, score_gold.long())
    p_score = torch.sigmoid(scores.squeeze(-1))
    loss_range = F.relu(p_score - torch.sigmoid(y_max)) + F.relu(torch.sigmoid(y_min) - p_score)
    loss_range = loss_range.mean()
    loss = loss_sup + loss_range
    return loss, loss_sup, loss_range