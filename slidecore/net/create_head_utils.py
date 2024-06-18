import slidecore
import torch.nn as nn

def create_head(arch:list=[], in_feats=None):
    ch = in_feats
    head_list = []
    for un in arch:
        if type(un) == int:
            lay = nn.Linear(in_features=ch, out_features=un)
            head_list.append(lay)
            head_list.append(nn.ReLU())
            ch = un
        elif un == 'A':
            lay = slidecore.attention.Attention(in_feats=ch, prob_flag=False)
            head_list.append(lay)
        elif un == 'AP':
            lay = slidecore.attention.Attention(in_feats=ch, prob_flag=True)
            head_list.append(lay)
    layers = nn.ModuleList(head_list)
    return layers, un
