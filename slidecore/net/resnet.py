import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import slidecore
import slidecore.net.create_head_utils
import random
import slidecore.net.calc_drv

relu = F.relu
relu6 = F.relu6
relu6_flg = False

def relu_act(X):
    if relu6_flg:
        Y = F.relu6(X)
    else:
        Y = F.relu(X)
    return Y

class SimpleUnit(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, output_relu=False):
        super(SimpleUnit,self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 nn.BatchNorm2d(num_features=out_ch),

                                 )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,X):
        Y = self.net(X) + X
        Y = self.pool(Y)
        return Y

class ResUnit(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, output_relu=False, **kwargs):
        super(ResUnit, self).__init__()
        self.scut = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.cnv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.cnv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)
        self.output_relu = output_relu

    def forward(self,X):
        XS = self.scut(X)
        X = self.cnv1(X)
        X = self.bn(X)
        X = F.relu(X)
        X = self.cnv2(X)
        X = self.bn2(X)
        Y = X + XS
        if self.output_relu:
            Y = F.relu(Y)
        return Y

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1,shortcut2=False, **args):
        super(Bottleneck, self).__init__()
        in_planes, planes=in_ch, out_ch
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        self.shortcut2 = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            if shortcut2:
                self.shortcut2 = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = relu_act(self.bn1(self.conv1(x)))
        out = relu_act(self.bn2(self.conv2(out)))
        if self.shortcut2:
            out += self.shortcut2(x)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu_act(out)
        return out


class SimpleHead(nn.Module):
    def __init__(self,in_feature=0, args=None):
        super(SimpleHead,self).__init__()
        head_list = []
        ch = in_feature
        head_arch = args['head_arch']
        out_cls = args['out_cls']

        self.head,un = slidecore.net.create_head_utils.create_head(arch=head_arch, in_feats=in_feature)
        self.lin = nn.Linear(in_features=un, out_features=out_cls)

    def forward(self,XX, target=None):
        for lay in self.head:
            XX = lay(XX)
        XX = self.lin(XX)
        if target is None:
            Y = F.softmax(XX, dim=1)
        else:
            Y = XX
        return Y
# ResNet architecture is defined as [(ch, num_repeats)]
# Head arch defined as
class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet,self).__init__()
        res_units = []
        self.args = args
        self.xsize=args['xsize']
        self.ysize=args['ysize']
        in_ch = args['in_ch']
        out_cls = args['out_cls']
        self.out_cls = out_cls
        res_unit_out_relu = args['res_unit_out_relu']
        arch = args['arch']
        head_arch = args['head_arch']
        resnet_50 = args['resnet50']
        shortcut2 = args.get('shortcut2', False)
        block = Bottleneck if resnet_50 else ResUnit
        self.to_tensor = transforms.ToTensor()
        self.std_flag = 'std' in args['hcf_list']
        self.log_var = 'LoG' in args['hcf_list']
        if self.log_var:
            in_ch += 1

        min_feat_map_size = args.get('min_feat_map_size', 2)
        self.dropout = None
        self.max_hcfs = []
        if args['dropout']:
            self.dropout = nn.Dropout2d()
        L = len(arch)
        xsize, ysize = self.xsize, self.ysize
        for k in range(L):
            ch,num_rep = arch[k]
            for n in range(num_rep):
                res_unit = block(in_ch=in_ch,out_ch=ch,output_relu=res_unit_out_relu, shortcut2=shortcut2)
                res_units.append(res_unit)
                in_ch = ch
            res_units.append(nn.MaxPool2d(kernel_size=2,stride=2))
            xsize//=2
            ysize//=2
            if xsize<=min_feat_map_size or ysize<=min_feat_map_size:
                break
        # This addition didnt really help we might consider to add smaller units
        # OR ignore it and perform average...
        while xsize>min_feat_map_size and ysize>min_feat_map_size:
            # ResUnit
            res_unit = SimpleUnit(in_ch=in_ch, out_ch=ch, output_relu=res_unit_out_relu)
            res_units.append(res_unit)
            in_ch = ch
            xsize //= 2
            ysize //= 2
        self.layers_list = nn.ModuleList(res_units)
        ch += self.std_flag
        ch += self.log_var
        self.arg = args
        if args['arc_cos_margin']>=0 and args['arc_cos_rad']>0:
            self.head = slidecore.arc_cos_head.ArcCosHead(in_feats=ch, args=args)
        else:
            self.head = SimpleHead(in_feature=ch, args=args)

    def set_max_hcf(self, max_hcf=[]):
        self.max_hcfs = max_hcf

    def get_model_args(self):
        return self.args
    # resize tensors and normalized them
    def norm(self,X):
        X = nn.functional.interpolate(X,size=(self.ysize, self.xsize), mode='bilinear')
        N, C, H, W = X.shape
        XR = X.view(N,C,W*H)
        min_vals = torch.mean(XR,dim=2, keepdim=True)
        max_vals = torch.std(XR,dim=2, keepdim=True)
        XR = (XR-min_vals)/(max_vals+1e-8)

        Y = XR.view(N,C,H,W)
        return Y

    # resize tensor to trained value
    def resize_ten(self,x, xsize=None, ysize=None):
        N,C,H,W = x.shape
        if xsize is None or ysize is None:
            xsize,ysize = self.xsize,self.ysize
        if H==ysize and W==xsize:
            return x
        y = F.interpolate(x, size=(ysize,xsize), mode='bilinear')
        return y

    def forward(self, X, target=None):
        if type(X) is not torch.Tensor or type(X) is np.ndarray:
            X = self.to_tensor(X)
            #X = torch.from_numpy(X)
        if self.std_flag:
            XV = X.view(X.size(0), -1)
            std_val = torch.std(XV, dim=1)
        if self.log_var:
            log_ten, log_var = slidecore.net.calc_drv.compute_LoG(X)
        # resize the tensor to what we have been train with
        #X = self.resize_ten(X)
        X = self.norm(X)
        if self.log_var:
            N,C,H,W = X.shape
            log_ten = log_ten.reshape((N, 1, H, W))
            X = torch.cat((X,log_ten), dim=1)
        for lay in self.layers_list:
            X = lay(X)
        N,C,H,W = X.shape
        # XV = X.view(N,C,W*H)
        # XX = torch.mean(XV,dim=2)
        if self.dropout:
            X = self.dropout(X)
        # Better way to pull
        XX=torch.nn.functional.adaptive_avg_pool2d(X, output_size=(1,1))
        XX=XX.view(N,C)
        # Runs head
        if self.std_flag:
            N,CH = XX.shape
            SV = std_val.view(N,1)
            if len(self.max_hcfs)>0:
                SV /= self.max_hcfs[0]
            XX = torch.cat((XX, SV), dim=1)
        if self.log_var:
            XX = torch.cat((XX,log_var), dim=1)
        Y = self.head(XX, target=target)
        return Y

    def infer(self,x, offset=50, inference_size=3):
        N,C,H,W = x.shape
        if H<self.ysize+offset or W<self.xsize+offset:
            x = self.resize_ten(x, xsize=self.xsize+offset, ysize=self.ysize+offset)
        y = None
        for k in range(inference_size):
            sx,sy=random.randint(0,offset),random.randint(0,offset)
            cur_x = x[:,:,sy:sy+self.ysize,sx:sx+self.xsize]
            cur_y = self(cur_x)
            if y is None:
                y = cur_y
            else:
                y += cur_y
        # Divide by inference size
        y = y/inference_size
        return y

    def save(self, file_path=None, optim=None, sched=None, epoch=-1):
        dct = {
            'args': self.args,
            'model_params' : self.state_dict(),
            'optim_params' : '' if optim is None else optim.state_dict(),
            'sched_params' : '' if sched is None else sched.state_dict(),
            'epoch':    epoch
        }
        torch.save(dct, file_path)
        return file_path

    @staticmethod
    def load(file_path=None):
        dct = torch.load(file_path, map_location=torch.device('cpu'))
        args = dct['args']
        resnet = ResNet(args=args)
        resnet.load_state_dict(dct['model_params'])
        optim_params = dct['optim_params']
        sched_params = dct['sched_params']
        epoch = dct['epoch']

        return resnet, args, optim_params,sched_params,epoch

########################## test part ####################
import argparse
def get_args():
    ap = argparse.ArgumentParser('train 1')
    ap.add_argument('--yaml_path', type=str, required=True, help="Full path of yaml file")
    args = ap.parse_args()
    yaml_obj = slidecore.yaml_obj.YamlObj(yaml_path=args.yaml_path)
    yaml_args = yaml_obj.get_params()
    return yaml_args

if __name__ == '__main__':
    X = np.random.randn(12,3,512,612)
    X = X.astype(np.float32)
    arch = [(18,1), (22,2), (32,1),(32,1),(32,1)]
    head_arch = [18]
    args = get_args()
    net = ResNet(args=args)
    Y = net(X)
    print(f'output.shape: {Y.shape}')
