import glob
import shutil

import slideapp.config
import utils.install_openslide
# Only at home run this
utils.install_openslide.add_openslide()
import torch
import slidecore
import slidecore.net.ensemble
from slidecore.net.ensemble import Ensemble as Ensemble
from slidecore.net.resnet import ResNet as ResNet
import slidecore.net.datatset
import slidecore.net.calc_drv
import numpy as np
import cv2
import torchvision
import os
import utils.extractor
import logging


class PredictImgs:
    def __init__(self, model_path=None, gpu=0,
                 ensemble_flag=False, inference_size=0,
                 cls_tile_thr=-1):
        if ensemble_flag:
            self.net = slidecore.net.ensemble.Ensemble.load(model_path=model_path)
        else:
            self.net,args, optim_params,sched_params,epoch = slidecore.net.resnet.ResNet.load(model_path)
        devstr = f'cuda:{gpu}' if gpu>=0 else 'cpu'
        logging.info(f'----> Working GPU:{devstr}')
        print(f'device: ->\t{devstr}')
        self.net = self.net.to(devstr)
        self.devstr = devstr
        self.inference_size = inference_size
        self.num_bad = 0
        self.cls_tile_thr = cls_tile_thr
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
                         out_dir=None,
                         file_exten='.jpeg',
                         batch_size=5, percentile=0.5,
                         write_tiles_flag=True,tiles_list=None,
                         tile_w=840, tile_h=840, n_tile_rows=0, n_tile_cols=0):
        #file_names = glob.glob(dir_path,file_exten)
        if tiles_list is None:
            file_names=PredictImgs.collect_files(dir_path, file_exten='jpg')
        else:
            file_names = [res[0] for res in tiles_list]
        if len(file_names) <=0:
            return False
        bad_dir,good_dir=None,None
        if write_tiles_flag:
            cur_dir = os.path.dirname(file_names[0]) if out_dir is None else out_dir
            shutil.rmtree(cur_dir,ignore_errors=True)
            bad_dir = os.path.join(cur_dir, 'bad_dir')
            good_dir = os.path.join(cur_dir, 'good_dir')
            os.makedirs(bad_dir, exist_ok=True)
            os.makedirs(good_dir, exist_ok=True)
        self.num_bad = 0
        pred_list = []
        ret_tiles_list = []
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
                cid = 1 if id == 1 else 0
                pr_bad = y_cur[kk, 1]
                if self.cls_tile_thr > 0: #and cid!= 1:
                    cid = 1 if pr_bad >= self.cls_tile_thr else 0
                cur_var = np.var(img_list[kk])
                if cur_var<=slideapp.config.tile_std_thr:
                    cid = 0
                pred_list.append(cid)
                if tiles_list is not None:
                    k_id = k + kk
                    cur_res = tiles_list[k_id]
                    cur_res = cur_res + (cid,)
                    tiles_list[k_id] = cur_res
                ret_tiles_list.append((file_names[k+kk], cid))
                if write_tiles_flag:
                    img_name = os.path.basename(file_names[k+kk])
                    cur_dir = bad_dir if cid==1 else good_dir
                    img_name = os.path.join(cur_dir, img_name)
                    cv2.imwrite(img_name, img_list[kk])
            k += len(img_list)
        pred_arr = np.array(pred_list)
        nones = np.sum(pred_arr>0)
        bad_p = nones/len(pred_list)
        slide_img = None
        if tiles_list is not None:
            slide_img = self.create_slide_img(pred_arr=pred_arr, tiles_list=tiles_list,
                                  tile_h=tile_h, tile_w=tile_w,
                                  n_tile_rows=n_tile_rows, n_tile_cols=n_tile_cols)
        return bad_p>=percentile, slide_img

    def create_slide_img(self,pred_arr=None, tiles_list=None, tile_h=0, tile_w=0, n_tile_rows=0, n_tile_cols=0):
        N = len(tiles_list)
        H = int((tile_h * n_tile_rows )/slideapp.config.slide_img_down_sample + 0.5)
        W = int((tile_w * n_tile_cols )/slideapp.config.slide_img_down_sample + 0.5)
        slide_img = np.zeros((H,W,3), np.uint8)
        tw,th = int(tile_w/slideapp.config.slide_img_down_sample), int(tile_h/slideapp.config.slide_img_down_sample)
        for k in range(N):
            fname,row,col,cid = tiles_list[k]
            img = cv2.imread(fname)
            img_ds = cv2.resize(img, (tw,th))
            cur_y,cur_x = row*th, col*tw
            slide_img[cur_y:cur_y+th, cur_x:cur_x+tw, :] = img_ds[0:cur_y+th, 0:cur_x+tw, :]
        # Next draw the rectanle
        red = (0,0,255)         # BGR
        green = (0,255,0)
        thickness = 8
        for k in range(N):
            fname,row,col,cid = tiles_list[k]
            if cid > 0:
                cur_y,cur_x = row*th, col*tw
                slide_img = cv2.rectangle(slide_img, (cur_x, cur_y), (cur_x+tw,cur_y+th), green, thickness=thickness)
        dirp = os.path.dirname(tiles_list[0][0])
        return slide_img




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

def collect_slides(root_dir, file_exten='ndpi',files_list_in=None):
    files_list = [] if files_list_in is None else files_list_in
    for dirpath, dirs, files in os.walk(root_dir):
        for filename in files:
            # should be 26
            if len(filename) < 26:
                continue
            fname = os.path.join(dirpath, filename)
            if fname.endswith(file_exten):
                files_list.append(fname)
        for dir in dirs:
            cur_dir = os.path.join(dirpath, dir)
            collect_slides(root_dir=cur_dir, file_exten=file_exten, files_list_in=files_list)
    return files_list

def work_on_slides(pred:PredictImgs=None, root_dir:str=None, file_exten='ndpi'):
    search_pat = os.path.join(root_dir, f'**{file_exten}')
    #file_names = glob.glob(search_pat, recursive=True)
    file_names = collect_slides(root_dir=root_dir, file_exten=file_exten)
    work_list=[]
    for fn in file_names:
        dir = os.path.dirname(fn)
        outputPath = os.path.join(dir, 'tiles')
        extractor = utils.extractor.TileExtractor(slide=fn, outputPath=outputPath, saveTiles=True)
        extractor.run()
        outputPath = extractor.tiles_dir
        pred.predict_from_dir(outputPath)
        work_list.append(outputPath)
        #del extractor
    print(f'work_list:\n{work_list}')

import argparse
def parse_args():
    ap = argparse.ArgumentParser('Ensemble')
    ap.add_argument('--model_path', type=str, required=True, help="Model path where to generate ensmeble")
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
