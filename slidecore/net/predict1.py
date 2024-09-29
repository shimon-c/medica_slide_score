import slidecore
import torch
import slidecore.net.datatset
from slidecore.net.train1 import compute_acc as compute_acc
import argparse
import slidecore.net.resnet
import sklearn
import matplotlib.pyplot as plt

def predict(args):
    model_path = args.model_path
    net,args, optim_params,sched_params,epoch = slidecore.net.resnet.ResNet.load(model_path)
    batch_size = 8
    net = net.to('cuda')
    test_ds = slidecore.net.datatset.DataSet(root_dir=args['test_set_dir'],
                                             good_path=args['test_good'], bad_path=args['test_bad'],
                                             xsize=args['xsize'], ysize=args['ysize'], test_flag=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    acc,conf_mat=compute_acc(net=net, loader=test_ld,calc_conf_mat=True)
    print(f'Final accuracy:{acc}, conf_mat:{conf_mat}')
    dmat = sklearn.metrics.ConfusionMatrixDisplay(conf_mat)
    dmat.plot()
    plt.show()

def parse_arg():
    ap = argparse.ArgumentParser('predict1')
    ap.add_argument('--model_path', type=str, required=True, help="Full model path")
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    predict(args)


