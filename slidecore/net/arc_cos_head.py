import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ArcCosHead(torch.nn.Module):
    def __init__(self, in_feats=0, num_hcfs=0, args=None):
        super(ArcCosHead, self).__init__()
        head_arch = args['head_arch']
        out_cls = args['out_cls']
        head_list = []
        ch = in_feats
        self.eps = 1e-6
        self.norm_eps = 1e-12
        self.rad = args['arc_cos_rad']
        self.margin = args['arc_cos_margin']
        for un in head_arch:
            lay = nn.Linear(in_features=ch, out_features=un)
            head_list.append(lay)
            head_list.append(nn.ReLU())
            ch = un
        
        self.layers = nn.ModuleList(head_list)
        self.lin = nn.Parameter(torch.Tensor(out_cls, un))
        in_sz = self.lin.size(0) * self.lin.size(1)
        std_frac = 2.
        stdv = math.sqrt(std_frac/in_sz)
        self.lin.data.uniform_(-stdv, stdv)
        print(f'arc-cos: margin:{self.margin}, rad:{self.rad}')

        
    def forward(self, X, target=None):
        for m in self.layers:
            X = m(X)

        #debug
        dbg = 0
        if dbg:
            y = X.matmul(self.lin.t())
            if target is None:
                y = F.softmax(y, dim=1)
            return y
        x_norm = F.normalize(X,dim=1, p=2, eps=self.norm_eps)
        lin_norm = F.normalize(self.lin, dim=1, p=2, eps=self.norm_eps)
        cos_t = x_norm.matmul(lin_norm.t()).clamp(min=self.eps, max=1-self.eps)
        # norm_lin = self.lin.norm(p=2,dim=1,keepdim=True)
        # self.lin.data = self.lin.data/norm_lin
        # cos_t = x_norm.matmul(self.lin.t()).clamp(min=self.eps, max=1 - self.eps)
        if target is not None:
            if self.margin>0:
                target = target.to(X.device)
                tar_view = target.view(target.size(0), -1)
                mask = torch.zeros_like(cos_t, dtype=torch.uint8, device=cos_t.device).scatter(1, tar_view,1)
                index = mask == 1
                cos_margin = torch.cos(torch.acos(cos_t) + self.margin)
                #cos_new = torch.where(mask, cos_margin, cos_t)
                cos_new = torch.where(index, cos_margin, cos_t)
                # ids = target>0
                # cos_n = torch.zeros_like(cos_t, device=cos_t.device, dtype=cos_t.dtype)
                # cos_n[:,:] = cos_t[:,:]
                # cos_n[ids] = cos_margin[ids]
            else:
                cos_new = cos_t
            y = cos_new * self.rad
        else:
            y = cos_t * self.rad
            y = F.softmax(y, dim=1)
        return y
        
