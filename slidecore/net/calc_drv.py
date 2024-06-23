import torch
import numpy as np
import cv2 as cv

def compute_vertical_drv(x, thr=0.2):
    N,C,H,W = x.shape
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

if __name__ == "__main__":

    def prepare_tens():
        sz = 6
        ten = torch.ones((2,3,sz,sz+4))
        ten[0,:,:,2:]=0
        ten[1,:,:,6:]=0
        drv_res = compute_vertical_drv(ten)
        print(drv_res)

    prepare_tens()



