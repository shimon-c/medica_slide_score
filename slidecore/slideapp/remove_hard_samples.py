import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from matplotlib.widgets import Button
import pandas as pd
import argparse

import sys

import slidecore.predict.predict_imgs
import slideapp
from slidecore.predict.predict_imgs import PredictImgs as PredictImgs
import os

def get_args():
    ap = argparse.ArgumentParser("Remove har samples")
    ap.add_argument("--csv_path", type=str, required=True, help="Dataset CSV")
    ap.add_argument("--classifier_path", type=str, required=True, help="classifier check point")
    args = ap.parse_args()
    return args

def get_csv(csv_path=None):
    df = pd.read_csv(csv_path)
    N, num_cols = df.shape
    return df

# Filter all results of classifier whihc have ben classified bu in range[0.5-thr,0.5+thr]
def filter_csv(df, thr=0.01, pred_path=None):
    pred = slidecore.predict.predict_imgs.PredictImgs(model_path=pred_path,
                                                                    cls_tile_thr=slideapp.config.classifer_tile_thr)
    N, num_cols = df.shape
    for k in range(N):
        img_path,cid = df.iloc[k,1], df.iloc[k,2]
        if cid<0:
            continue
        img = cv2.imread(img_path)
        img_list = [img,]
        y = pred.predict(img_list)
        pr = y[0,1]
        if pr>(0.5-thr) and (pr<0.5+thr):
            df.iloc[k,2] = -1

if __name__ == '__main__':
    args = get_args()
    df = get_csv(args.csv_path)
    filter_csv(df, pred_path=args.classifier_path)
    dir, csv_name = os.path.dirname(args.csv_path), os.path.basename(args.csv_path)
    csv_name_list = csv_name.split('.csv')
    csv_name = f'{csv_name_list[0]}_cleared.csv'
    csv_name = os.path.join(dir, csv_name)
    df.to_csv(csv_name)