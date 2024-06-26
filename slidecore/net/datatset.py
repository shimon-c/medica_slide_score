import torch
from torch.utils.data import Dataset as TorchDataset
import cv2
import os
import matplotlib.pyplot as plt
import random
import numpy as np
import torchvision.transforms as transforms
# RandomResizedCrop
# A.CLAHE(p=0.1),
#         A.Posterize(p=0.1),
#         A.ToGray(p=0.1),
#         A.ChannelShuffle(p=0.05),
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur,Resize,RandomScale , HorizontalFlip ,
    RandomBrightnessContrast, Flip, OneOf, Compose, CLAHE, Posterize
)

class CutOff(ImageOnlyTransform):
    def __init__(self):
        super(CutOff, self).__init__()
        self.cutoff_size = 20

    def apply(self, img, **params):
        H,W,C = img.shape
        x,y,sz = random.randint(W-self.cutoff_size), random.randint(H-self.cutoff_size), random.randint(self.cutoff_size)
        img[y:y+sz,x:x+sz,:] = 0
        return img

#dataset for good/bad
class DataSet(TorchDataset):
    GOOD_IMG=0
    NOT_RELV_IMG=2   #1
    BAD_IMG=1        #2
    def __init__(self, root_dir=None,
                 good_path:str='', bad_path:str=None, not_rel:str=None, end_str='.jpeg',
                 xsize=512, ysize=512, test_flag=False, augmentations=[]):
        super(DataSet,self).__init__()
        assert root_dir is not None
        assert good_path!='' and good_path is not None
        good_path = os.path.join(root_dir, good_path)
        bad_path = os.path.join(root_dir, bad_path) if bad_path is not None else None
        not_rel = None if not_rel is None else os.path.join(root_dir, not_rel)
        self.to_tensor = transforms.ToTensor()
        self.good_images = []
        self.bad_images = []
        self.not_rel_images = []
        self.all_imgs=[]
        self.exten = end_str
        self.xsize,self.ysize = xsize,ysize
        self.good_images = self.load_images_from_root_dir(root_dir=good_path, names_list=self.good_images)
        self.augs = None
        self.max_std = 1
        self.resize_obj = Resize (ysize, xsize, interpolation=1, always_apply=False, p=1)
        if not test_flag:
            aug_list = []

            for aug in augmentations:
                if 'Blur' in aug:
                    aug_list.append(OneOf([
                        # MotionBlur(p=0.2),
                        MedianBlur(blur_limit=3, p=0.1),
                        Blur(blur_limit=3, p=0.1),
                    ], p=0.2), )
                elif 'GaussNoise' in aug:
                    aug_list.append(GaussNoise())
                elif 'OpticalDistortion' in aug:
                    aug_list.append(OpticalDistortion())
                elif 'RandomScale' in aug:
                    aug_list.append(RandomScale())
                elif 'Flip' in aug:
                    aug_list.append(Flip())
                elif 'HorizFlip' in aug:
                    aug_list.append(HorizontalFlip(p=0.5))
                elif 'CLAHE' in aug:
                    aug_list.append(A.CLAHE(p=0.1))
                elif 'Posterize' in aug:
                    aug_list.append(A.Posterize(p=0.1))
                elif 'Hue' in aug:
                    aug_list.append(A.HueSaturationValue())
                elif 'CutOff' in aug:
                    aug_list.append(CutOff())
            print(f'dataset #augs: {len(aug_list)}')
            # I think this augmentation is problematic I need here just resize
            #aug_list.append(A.Resize(height=ysize, width=xsize))
            aug_list.append(A.RandomResizedCrop(height=ysize, width=xsize))
            self.augs = Compose(aug_list)
        for img in self.good_images:
            self.all_imgs.append((img, DataSet.GOOD_IMG))
        if bad_path is not None:
            self.bad_images = self.load_images_from_root_dir(root_dir=bad_path, names_list=self.bad_images)
        for img in self.bad_images:
            self.all_imgs.append((img, DataSet.BAD_IMG))
        if not_rel is not None:
            self.not_rel_images = self.load_images_from_root_dir(root_dir=not_rel, names_list=self.not_rel_images)
        for img in self.not_rel_images:
            self.all_imgs.append((img, DataSet.NOT_RELV_IMG))
        self.clear_low_std_imgs()
        # Compute number of classes
        self.cls_num = 0
        for tp in self.all_imgs:
            self.cls_num = max(self.cls_num, tp[1])
        self.cls_num += 1
        self.dataset_stat_str = f'bad_imgs:{len(self.bad_images)}, good_imgs:{len(self.good_images)}, not_rel:{len(self.not_rel_images)}'
        print(self.dataset_stat_str)


    def load_images_from_root_dir(self, root_dir:str=None, names_list=[]):
        for root, dirs, files in os.walk(root_dir, topdown=False):
            for name in files:
                cur_name = os.path.join(root, name)
                if os.path.isfile(cur_name) and cur_name.endswith(self.exten):
                    names_list.append(cur_name)

            for dir in dirs:
                cur_root_dir = os.path.join(root_dir,dir)
                names_list = self.load_images_from_root_dir(root_dir=cur_root_dir, names_list=names_list)
        names_set =  set(names_list)
        names_list = list(names_set)
        return names_list

    def __getitem__(self, item):
        item = item%len(self.all_imgs)
        ent = self.all_imgs[item]
        img = cv2.imread(ent[0])
        lab = torch.zeros(1, dtype=torch.int64)
        lab[0] = ent[1]
        #img = cv2.resize(img, (self.xsize, self.ysize))
        if self.augs is not None:
            img = self.augs(image=img)['image']
        else:
            img = self.resize_obj(image=img)['image']
        img = img.astype(np.float32)
        img_ten = self.to_tensor(img)

        return img_ten,lab

    def clear_low_std_imgs(self, min_std=10):
        imgs = []
        self.max_std = 0
        L = len(self.all_imgs)
        for k in range(L):
            img_ten,lab = self[k]
            cur_std = torch.std(img_ten)
            self.max_std = max(cur_std,self.max_std)
            if cur_std>min_std:
                ent = self.all_imgs[k]
                imgs.append((ent[0], ent[1]))
        self.all_imgs = imgs
        print(f'cleared images={L-len(imgs)}')
    def __len__(self):
        return len(self.all_imgs)

    def shuffle(self):
        random.shuffle(self.all_imgs)

    def get_num_of_classes(self):
        return self.cls_num


############################ Test ###############
good_path = r"C:\Users\shimon.cohen\data\medica\imgdb\imgdb\train_set\GoodFocus"
bad_path = r"C:\Users\shimon.cohen\data\medica\imgdb\imgdb\train_set\BadFocus"
not_rel=r"C:\Users\shimon.cohen\data\medica\imgdb\imgdb\train_set\NotRelevant"

if __name__ == '__main__':
    data = DataSet(good_path=good_path,bad_path=bad_path,not_rel=not_rel)
    data.shuffle()
    ds_size = len(data)
    print(f'Datsize:{ds_size}')
    lab_str = ['Good', 'NotRel', 'Bad']
    for k in range(ds_size):
        img,lab = data[k]
        plt.title(lab_str[lab])
        plt.imshow(img)
        key = input(f"Image: {k}, Enter and key:")

