import torch
import torch.nn as nn
import os
import glob
import slidecore


class Ensemble(nn.Module):
    def __init__(self, model_path:str=None):
        super(Ensemble, self).__init__()
        if model_path is None:
            return
        dir = os.path.dirname(model_path)
        sreach_str = os.path.join(dir, f'*.pt')
        model_list = glob.glob(sreach_str)
        resnets = []
        self.num_classes = None

        for chk_pnt in model_list:
            resnet,args, optim_params,sched_params,epoch = slidecore.resnet.ResNet.load(chk_pnt)
            resnets.append(resnet)
        self.models = nn.ModuleList(resnets)

    def forward(self, x):
        y = None
        for m in self.models:
            cur_y = m(x)
            if y is None:
                y = cur_y
            else:
                y += cur_y
        y = y / len(self.models)
        return y

    # Make inference default 50 pixels, and size=3
    def infer(self, x, offset=50, inference_size=3):
        y = None
        for m in self.models:
            cur_y = m.infer(x, offset=offset, inference_size=inference_size)
            if y is None:
                y = cur_y
            else:
                y += cur_y
        y = y / len(self.models)
        return y

    def get_model_args(self):
        args = self.models[0].get_model_args()
        return args

    def save(self, dir:str=''):
        fname = os.path.join(dir, 'ensemble.pt')
        args_list = [(m.args,m.state_dict()) for m in self.models]
        dct = {
            'model_params': args_list,
        }
        torch.save(dct, fname)
        return fname


    @staticmethod
    def load(model_path):
        dct = torch.load(model_path, map_location=torch.device('cpu'))
        args = dct['model_params']
        n = len(args)
        net_list = []
        for margs, stat_dict in args:
            resnet = slidecore.resnet.ResNet(args=margs)
            resnet.load_state_dict(state_dict=stat_dict)
            net_list.append(resnet)
        ens = Ensemble(model_path=None)
        ens.models = nn.ModuleList(net_list)
        return ens

import argparse
def parse_args():
    ap = argparse.ArgumentParser('Ensemble')
    ap.add_argument('--model_path', type=str, required=True, help="Mode path where to generate ensmeble")
    ap.add_argument('--test_ensemble', type=bool, default=True, help="Performs test on full test images")
    args = ap.parse_args()
    return args

import slidecore.net.train1
if __name__ == '__main__':
    args = parse_args()
    ens = Ensemble(model_path=args.model_path)
    ens_model_path = os.path.join(os.path.dirname(args.model_path), 'ensemble_model')
    os.makedirs(ens_model_path, exist_ok=True)
    print(f'output-dir:{ens_model_path}')
    model_name = ens.save(ens_model_path)
    print(f'Ensemble saved: {model_name}')
    if args.test_ensemble :
        ens = Ensemble.load(model_path=model_name)
        print(f'Enemsble len:{len(ens.models)}')
        model_args = ens.models[0].args
        xsize,ysize = model_args['xsize'], model_args['ysize']
        test_good, test_bad=model_args['test_good'], model_args['test_bad']
        test_dir = model_args['test_set_dir']
        test_ds = slidecore.datatset.DataSet(root_dir=test_dir,
                                             good_path=test_good, bad_path=test_bad,
                                             xsize=xsize, ysize=ysize, test_flag=True)
        test_ld = torch.utils.data.DataLoader(test_ds, batch_size=6, shuffle=False)
        ens = ens.to('cuda')
        acc, conf_mat = slidecore.net.train1.compute_acc(net=ens, loader=test_ld, calc_conf_mat=True)
        print(f'acc={acc}\n conf_mat:\n{conf_mat}')

