import glob
import utils.install_openslide
utils.install_openslide.add_openslide()
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
import utils.extractor


class PredictImgs:
    def __init__(self, model_path=None, gpu=0, ensemble_flag=False, inference_size=0):
        if ensemble_flag:
            self.net = slidecore.net.ensemble.Ensemble.load(model_path=model_path)
        else:
            self.net,args, optim_params,sched_params,epoch = slidecore.net.resnet.ResNet.load(model_path)
        devstr = devstr = f'cuda:{gpu}' if gpu>=0 else 'cpu'
        self.net = self.net.to(devstr)
        self.devstr = devstr
        self.inference_size = inference_size
        self.num_bad = 0
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
            dsize = (W,H)
            ten = torch.zeros((N,C,H,W), dtype=torch.float32)
            for k in range(N):
                img = imgs_list[k].astype(np.float32)
                if img.shape[1]!=H or img.shape[2] != W:
                    img = cv2.resize(img,dsize=dsize, interpolation = cv2.INTER_LINEAR)
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

    @staticmethod
    def collect_files(root_dir, file_exten='jpg'):
        files_list = []
        for dirpath, dirs, files in os.walk(root_dir):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                if fname.endswith(file_exten):
                    files_list.append(fname)
        return files_list

    @torch.no_grad()
    def predict_from_dir(self, dir_path:str=None,
                         file_exten='.jpeg',
                         batch_size=5, percentile=0.5,
                         write_tiles_flag=True):
        #file_names = glob.glob(dir_path,file_exten)
        file_names=PredictImgs.collect_files(dir_path, file_exten='jpg')
        if len(file_names) <=0:
            return False
        bad_dir,good_dir=None,None
        if write_tiles_flag:
            cur_dir = os.path.dirname(file_names[0])
            bad_dir = os.path.join(cur_dir, 'bad_dir')
            good_dir = os.path.join(cur_dir, 'good_path')
            os.makedirs(bad_dir, exist_ok=True)
            os.makedirs(good_dir, exist_ok=True)
        self.num_bad = 0
        pred_list = []
        N = len(file_names)
        k = 0
        while k<N:
            img_list = []
            kk = k
            while kk<N and len(img_list)<batch_size:
                img = cv2.imread(file_names[kk])
                img_list.append(img)
                kk += 1
            y_cur = self.predict(img_list)

            for kk in range(len(img_list)):
                id = np.argmax(y_cur[kk,:])
                cid = 1 if id==1 else 0
                pred_list.append(cid)
                if write_tiles_flag:
                    img_name = os.path.basename(file_names[k+kk])
                    cur_dir = bad_dir if cid==1 else good_dir
                    img_name = os.path.join(cur_dir, img_name)
                    cv2.imwrite(img_name, img_list[kk])
            k += len(img_list)
        pred_arr = np.array(pred_list)
        nones = np.sum(pred_arr>0)
        bad_p = nones/len(pred_list)
        return bad_p>=percentile



# Ability to test full work


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
    # there is a problem of the initial size
    xsize,ysize=380,380
    test_ds = slidecore.net.datatset.DataSet(root_dir=test_dir,
                                         good_path=test_good, bad_path=test_bad,
                                         xsize=xsize, ysize=ysize, test_flag=True)
    batch_size = 6
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    device = 'cuda:0'
    cls = cls.to(device)
    acc, conf_mat = slidecore.net.train1.compute_acc(net=cls, loader=test_ld, calc_conf_mat=True, device=device)
    print(f'acc={acc}\n conf_mat:\n{conf_mat}')

def collect_slides(root_dir, file_exten='ndpi'):
    files_list = []
    for dirpath, dirs, files in os.walk(root_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(file_exten):
                files_list.append(fname)
    return files_list

def work_on_slides(pred:PredictImgs=None, root_dir:str=None, file_exten='ndpi'):
    search_pat = os.path.join(root_dir, f'**{file_exten}')
    #file_names = glob.glob(search_pat, recursive=True)
    file_names = collect_slides(root_dir=root_dir, file_exten=file_exten)
    for fn in file_names:
        dir = os.path.dirname(fn)
        outputPath = os.path.join(dir, 'tiles')
        extractor = utils.extractor.TileExtractor(slide=fn, outputPath=outputPath, saveTiles=True)
        extractor.run()
        pred.predict_from_dir(outputPath)
        del extractor

import argparse
def parse_args():
    ap = argparse.ArgumentParser('Ensemble')
    ap.add_argument('--model_path', type=str, required=True, help="Mode path where to generate ensmeble")
    ap.add_argument('--inference_size', type=int, default=0, help="perform ineference")
    ap.add_argument('--slides_dir', type=str, default="", help="directory of slides")
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    im1=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\GoodFocus\GoodFocus_ANONJSBHSI1F2_1_1_level_17_size_840\4_19.jpeg"
    # im2 doesnt look like a bad focus....
    im2=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\BadFocus\BadFocus_ANONFACHSI1RE_4_1_level_17_size_840\9_20.jpeg"
    im3=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\BadFocus\BadFocus_ANONFACHSI1RE_4_1_level_17_size_840\10_22.jpeg"
    model_path=r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_6_acc_92.pt"
    model_path = r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_12_0.921743.pt"
    model_path = r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_14_acc_93.pt"
    model_path=r"C:\Users\shimon.cohen\PycharmProjects\new_slidecore\model\output_model\resnet_epoch_17_0.924198.pt"
    pi = PredictImgs(model_path=model_path)
    args = parse_args()
    if args.slides_dir != '':
        work_on_slides(pred=pi, root_dir=args.slides_dir)
    img1 = cv2.imread(im1)
    img2 = cv2.imread(im2)
    img3 = cv2.imread(im3)
    imgs = [img1, img2, img3]
    res = pi.predict(imgs)
    print(f'res={res}')

    res = pi.predict(imgs, inference_size=3)
    print(f'inference_res={res}')

    full_test()
