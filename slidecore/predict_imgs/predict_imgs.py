import torch
import slidecore
import slidecore.net.ensemble
from slidecore.net.ensemble import Ensemble as Ensemble
from slidecore.net.resnet import ResNet as ResNet
import slidecore.net.datatset
import numpy as np
import cv2
import torchvision
import os


class PredictImgs:
    def __init__(self, model_path=None, gpu=0, ensemble_flag=False, inference_size=0):
        if ensemble_flag:
            self.net = slidecore.net.ensemble.Ensemble(model_path=model_path)
        else:
            self.net,args, optim_params,sched_params,epoch = slidecore.net.resnet.ResNet.load(model_path)
        devstr = devstr = f'cuda:{gpu}' if gpu>=0 else 'cpu'
        self.net = self.net.to(devstr)
        self.devstr = devstr
        self.inference_size = inference_size
        self.net.eval()
        self.to_tensor = torchvision.transforms.ToTensor()

    def get_model_args(self):
        return self.net.get_model_args()

    def to(self,dev):
        self.net = self.net.to(dev)
        return self

    def eval(self):
        self.net.eval()

    def __call__(self, x, *args, **kwargs):
        if self.inference_size<=0:
            return self.net(x)
        else:
            return self.predict(imgs_list=x, inference_size=self.inference_size)

    @torch.no_grad()
    def predict(self, imgs_list:list=[], inference_size=False):
        return_ten = False
        if type(imgs_list) is list:
            N = len(imgs_list)
            H,W,C = imgs_list[0].shape
            ten = torch.zeros((N,C,H,W), dtype=torch.float32)
            for k in range(N):
                img = imgs_list[k].astype(np.float32)
                img = self.to_tensor(img)
                ten[k,...] = img
        else:
            ten = imgs_list
            return_ten = True
        # move to device
        x = ten.to(self.devstr)
        if inference_size>0:
            y = self.net.infer(x, inference_size=inference_size)
        else:
            y = self.net(x)
        if return_ten:
            return y
        y_np = y.cpu().detach().numpy()
        return y_np

# Ability to test full work
import argparse
def parse_args():
    ap = argparse.ArgumentParser('Ensemble')
    ap.add_argument('--model_path', type=str, required=True, help="Mode path where to generate ensmeble")
    ap.add_argument('--inference_size', type=int, default=0, help="perform ineference")
    args = ap.parse_args()
    return args

import slidecore.net.train1
def full_test():
    args = parse_args()
    ensemble_flag = True if 'ensemble' in args.model_path else False
    cls = PredictImgs(model_path=args.model_path, ensemble_flag=ensemble_flag, inference_size=args.inference_size)
    out_model_path = os.path.join(os.path.dirname(args.model_path), 'output_model')
    os.makedirs(out_model_path, exist_ok=True)
    print(f'output-dir:{out_model_path}')
    model_args = cls.get_model_args()
    xsize,ysize = model_args['xsize'], model_args['ysize']
    test_good, test_bad=model_args['test_good'], model_args['test_bad']
    test_dir = model_args['test_set_dir']
    test_ds = slidecore.net.datatset.DataSet(root_dir=test_dir,
                                         good_path=test_good, bad_path=test_bad,
                                         xsize=xsize, ysize=ysize, test_flag=True)
    batch_size = 12
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    cls = cls.to('cuda')
    acc, conf_mat = slidecore.net.train1.compute_acc(net=cls, loader=test_ld, calc_conf_mat=True)
    print(f'acc={acc}\n conf_mat:\n{conf_mat}')

if __name__ == '__main__':
    im1=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\GoodFocus\GoodFocus_ANONJSBHSI1F2_1_1_level_17_size_840\4_19.jpeg"
    # im2 doesnt look like a bad focus....
    im2=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\BadFocus\BadFocus_ANONFACHSI1RE_4_1_level_17_size_840\9_20.jpeg"
    im3=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\BadFocus\BadFocus_ANONFACHSI1RE_4_1_level_17_size_840\10_22.jpeg"
    model_path=r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_6_acc_92.pt"
    model_path = r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_12_0.921743.pt"
    model_path = r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_14_acc_93.pt"
    model_path=r"C:\Users\shimon.cohen\PycharmProjects\new_slidecore\model\resnet_epoch_26_0.919474.pt"
    pi = PredictImgs(model_path=model_path)
    img1 = cv2.imread(im1)
    img2 = cv2.imread(im2)
    img3 = cv2.imread(im3)
    imgs = [img1, img2, img3]
    res = pi.predict(imgs)
    print(f'res={res}')

    res = pi.predict(imgs, inference_size=3)
    print(f'inference_res={res}')

    full_test()
