import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


def compute_sig_to_noise(img=None, xo=-1,yo=-1,size=50):
    img_v = img[yo-size:yo+size, xo-size:xo+size,:]
    mval = np.mean(img_v)
    sval = np.std(img_v)
    return mval, sval,mval/sval

def parse_args():
    ap = argparse.ArgumentParser('Signal To Noise')
    ap.add_argument('--image_path', type=str, required=True, help="Full path to image")
    ap.add_argument('--image_center', type=str, required=True, help="Center of area")
    ap.add_argument('--size', type=int, default=50, help="Size to compute SNR")
    args = ap.parse_args()
    args.image_center = args.image_center.split(',')
    args.image_center[0] = int(args.image_center[0])
    args.image_center[1] = int(args.image_center[1])
    return args

if __name__ == '__main__':
    args = parse_args()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if args.image_center[0]<=0 or args.image_center[1]<=0:
        ax.imshow(img)
        plt.show()
    else:
        mval, sval, snr = compute_sig_to_noise(img,xo=args.image_center[0], yo=args.image_center[1], size=args.size)
        print(f'signal={mval}, noise={sval}, SNR={snr}')