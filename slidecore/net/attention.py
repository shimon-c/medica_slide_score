import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_feats=None, squeeze_factor=4, prob_flag=False):
        super(Attention,self).__init__()
        md_feats = in_feats//squeeze_factor
        self.net = nn.Sequential(nn.Linear(in_features=in_feats, out_features=md_feats),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(num_features=md_feats),
                                 nn.Linear(in_features=md_feats, out_features=in_feats))
        self.prob = None
        if prob_flag:
            self.prob = nn.Linear(in_features=in_feats,out_features=in_feats)

    def forward(self, X):
        Y1 = self.net(X)
        if self.prob is not None:
            y2 = self.prob(X)
            Y1 = Y1 * torch.sigmoid(y2)
        return Y1