import torch
import numpy as np
import cv2 as cv

def compute_vertical_drv(x, thr=0.2):
    N,C,H,W = x.shape
    if x.dtype != torch.float32:
        x = x.type(torch.float32)
    # Reduce the number of channels, another option to  take the red channel
    x = torch.mean(x, dim=1)
    z_x = torch.zeros_like(x)
    drv = x[:,:,1:] - x[:,:,0:W-1]
    drv = torch.abs(drv)
    z_x[:,:,:W-1] = drv
    drv = z_x
    max_val, max_ids = torch.max(drv,dim=2, keepdim=True)
    drv = drv / max_val
    drv_ids = drv>=thr
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    drv_val = torch.where(drv_ids>0, ones, zeros)
    drv_res = torch.mean(drv_val,dim=1)
    return drv_res

def get_prominent_lines(drv, k_pnts=2):
    max_ids = torch.argsort(drv)
    max_ids = max_ids[0, -k_pnts:].tolist()
    return max_ids


import argparse
def parse_args():
    ap = argparse.ArgumentParser('DRV')
    ap.add_argument('--image_path', type=str, required=True, help="Path to an image")
    args = ap.parse_args()
    return args

def vertical_on_image(img_path:str=None):
    img = cv.imread(img_path)
    H,W,C = img.shape
    ten = torch.from_numpy(img)
    ten = torch.permute(ten, (2,0,1))
    ten = ten.reshape((1,C,H,W))
    drv = compute_vertical_drv(ten)

    max_ids = get_prominent_lines(drv)
    for id in max_ids:
        x = id
        start_point, end_point = (x,0),(x,H)
        color = (0, 255, 0)
        img = cv.line(img, start_point, end_point, color=color, thickness=2)
    filename = f'{img_path}_line.jpeg'
    cv.imwrite(filename, img)
    print(f'imgae:{filename}')

if __name__ == "__main__":


    def prepare_tens():
        sz = 6
        ten = torch.ones((2,3,sz,sz+4))
        ten[0,:,:,2:]=0
        ten[1,:,:,6:]=0
        drv_res = compute_vertical_drv(ten)
        print(drv_res)

    prepare_tens()
    args = parse_args()
    vertical_on_image(img_path=args.image_path)



