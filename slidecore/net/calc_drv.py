import torch
import numpy as np
import cv2 as cv2


def compute_dx(x):
    N, C, H, W = x.shape
    if x.dtype != torch.float32:
        x = x.type(torch.float32)
    # Reduce the number of channels, another option to  take the red channel
    x = torch.mean(x, dim=1)
    z_x = torch.zeros_like(x)
    drv = x[:, :, 1:] - x[:, :, 0:W - 1]
    drv = torch.abs(drv)
    z_x[:, :, :W - 1] = drv
    drv = z_x
    return drv

def compute_dy(x):
    N, C, H, W = x.shape
    if x.dtype != torch.float32:
        x = x.type(torch.float32)
    # Reduce the number of channels, another option to  take the red channel
    x = torch.mean(x, dim=1)
    z_x = torch.zeros_like(x)
    drv = x[:, 1:, :] - x[:, 0:H - 1, :]
    drv = torch.abs(drv)
    z_x[:, :H - 1, :] = drv
    drv = z_x
    return drv

def compute_LoG(x, LoG_thr=10):
    N, C, H, W = x.shape
    if x.dtype != torch.float32:
        x = x.type(torch.float32)
    x = torch.mean(x, dim=1)
    z_x = torch.zeros_like(x)

    z_x[:,1:-1,1:-1] = -x[:,1:-1,1:-1]*4 + x[:,0:-2,1:-1] + x[:,2:,1:-1] + x[:,1:-1,2:] + x[:,1:-1,0:-2]
    # zids = z_x<=LoG_thr
    # z_x[zids] = 0

    z_v = z_x.view((N,W*H))
    # oids = z_v > LoG_thr
    # var_z = torch.std(z_v[oids], dim=1)
    var_z = torch.std(z_v, dim=1,keepdim=True)
    var_z = var_z/(4*255)
    return z_x, var_z


def compute_grad(x, grad_thr=10):
    dy = compute_dy(x)
    dx = compute_dx(x)
    grad = dx + dy
    ids = grad <= grad_thr
    return grad

def compute_vertical_drv(x, thr=0.2):
    drv = compute_dx(x)
    max_val, max_ids = torch.max(drv,dim=2, keepdim=True)
    max_val = 255
    drv = drv / max_val
    drv_ids = drv>=thr
    ones = torch.ones_like(drv)
    zeros = torch.zeros_like(drv)
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


import matplotlib.pyplot as plt
def vertical_on_image(img_path:str=None):
    img = cv2.imread(img_path)
    H,W,C = img.shape
    ten = torch.from_numpy(img)
    ten = torch.permute(ten, (2,0,1))
    ten = ten.reshape((1,C,H,W))
    thr = 0.2
    drv = compute_vertical_drv(ten, thr=thr)

    max_ids = get_prominent_lines(drv)
    for id in max_ids:
        if drv[0,id] < thr: continue
        x = id
        start_point, end_point = (x,0),(x,H)
        color = (0, 255, 0)
        img = cv2.line(img, start_point, end_point, color=color, thickness=2)
    filename = f'{img_path}_line.jpeg'
    #cv2.imwrite(filename, img)
    print(f'image:{filename}')
    grd = compute_grad(ten)
    grd = torch.permute(grd, (1,2,0))
    grad = grd.numpy()
    filename = f'{img_path}_grad.jpeg'
    #cv2.imwrite(filename, grad)
    print(f'image-grad:{filename}')
    z_x, v_z = compute_LoG(ten)
    print(f'LoG var: {v_z}')
    z_x = torch.permute(z_x, (1, 2, 0))
    LoG = z_x.numpy()
    filename = f'{img_path}_LoG.jpeg'
    #cv2.imwrite(filename, LoG)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1,3,2)
    grad = cv2.cvtColor(grad,cv2.COLOR_GRAY2BGR)
    plt.imshow(grad)
    plt.subplot(1,3,3)
    LoG = cv2.cvtColor(LoG, cv2.COLOR_GRAY2BGR)
    plt.imshow(LoG, cmap='gray')
    plt.show()

if __name__ == "__main__":


    def prepare_tens():
        sz = 6
        ten = torch.ones((2,3,sz,sz+4))
        ten[0,:,:,2:]=0
        ten[1,:,:,6:]=0
        drv_res = compute_vertical_drv(ten)
        print(drv_res)

    #prepare_tens()
    args = parse_args()
    vertical_on_image(img_path=args.image_path)



