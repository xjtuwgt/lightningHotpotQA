import torch
from torch import Tensor as T
from torch import nn


class RangeModel(nn.Module):
    def __init__(self, args):
        super(RangeModel, self).__init__()
        self.args = args
        self.cls_emb_dim = self.args.cls_emb_dim
        self.emb_dim = self.args.emb_dim
        self.score_dim = self.emb_dim - self.cls_emb_dim
        self.hid_dim = self.args.hid_dim




    def forward(self, x: T):
        assert x.shape[1] == self.emb_dim
        cls_x = x[:,:self.cls_emb_dim]
        score_x = x[:,self.cls_emb_dim:]

        return