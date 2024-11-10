import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from matplotlib.widgets import Button
import pandas as pd
import argparse

def get_cmd_args():
    ap = argparse.ArgumentParser("TilesUi")
    ap.add_argument('--root_dir', type=str, required=True, help="Root directory of slides")
    args = ap.parse_args()
    return args

args = get_cmd_args()
fig, ax = plt.subplots(nrows=1, ncols=1)
# set the size of the figure
figsize=(8, 10)
fig.subplots_adjust(bottom=0.2)
fig.set_size_inches(figsize[0],figsize[1])


img1_path=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg"
img2_path=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_4_11.jpeg"
class Index:
    def __init__(self, root_dir, wildcard='*.jpg'):
        # dir = os.path.join(root_dir, f'**/{wildcard}')
        # file_names = glob.glob(dir)
        file_names = self.collect_files(root_dir)
        self.img_list = [[fn,-1] for fn in file_names]
        self.ind = 0
        self.root_dir = root_dir
        self.cur_img_name = None
        self.show_current_image()

    def collect_files(self, root_dir, file_exten='jpg'):
        files_list = []
        for dirpath, dirs, files in os.walk(root_dir):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                if fname.endswith(file_exten):
                    files_list.append(fname)
        return files_list

    def show_current_image(self):
        self.ind = self.ind % len(self.img_list)
        img_path = self.img_list[self.ind][0]
        self.cur_img_name = img_path
        img = cv2.imread(img_path)
        ax.imshow(img)
        plt.draw()

    def next(self, event):
        self.ind += 1
        self.show_current_image()

    def prev(self, event):
        self.ind -= 1
        self.show_current_image()

    def ok(self, event):
        print('ok')
        self.img_list[self.ind][1] = 1

    def bad(self, event):
        print('bad')
        self.img_list[self.ind][1] = 0

    def save(self,event):
        print('save')
        file_name = os.path.join(self.root_dir, 'good_bad_train.csv')
        df = pd.DataFrame(data=self.img_list, columns=["file_name", "class"])
        df.to_csv(file_name)
        print(f'saved file: {file_name}')

root_dir=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set"
print(f'working on directory:{args.root_dir}')
callback = Index(root_dir=args.root_dir)
ax_ok = fig.add_axes([0.1, 0.05, 0.1, 0.075])
ax_bad = fig.add_axes([0.21, 0.05, 0.1, 0.075])
ax_save = fig.add_axes([0.35, 0.05, 0.1, 0.075])
axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])

bok = Button(ax_ok, 'OK')
bok.on_clicked(callback.ok)
bbad = Button(ax_bad, 'BAD')
bbad.on_clicked(callback.bad)
save_b = Button(ax_save, 'Save')
save_b.on_clicked(callback.save)

bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()