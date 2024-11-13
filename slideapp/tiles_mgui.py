import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from matplotlib.widgets import Button
import pandas as pd
import argparse

# Remember to change files exten to ".jpg" on linux machine
def get_cmd_args():
    ap = argparse.ArgumentParser("TilesUi")
    ap.add_argument('--root_dir', type=str, required=True, help="Root directory of slides")
    ap.add_argument('--csv_file', type=str, required=None, help="path to previous csv")
    ap.add_argument('--file_exten', type=str, default="jpeg", help="path to previous csv")
    ap.add_argument('--skip_std', type=float, default=20, help="Value of variance to skip")
    args = ap.parse_args()
    return args

args = get_cmd_args()
fig, ax = plt.subplots(nrows=1, ncols=1)
# set the size of the figure
figsize=(8, 10)
fig.subplots_adjust(bottom=0.2)
fig.set_size_inches(figsize[0],figsize[1])

ax_txt = None
txt_btn = None
img1_path=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg"
img2_path=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_4_11.jpeg"
class Index:
    def __init__(self, root_dir, csv_path=None, wildcard='*.jpg', skip_std=-1):
        # dir = os.path.join(root_dir, f'**/{wildcard}')
        # file_names = glob.glob(dir)
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            N,num_cols = df.shape
            file_names = []
            self.ind = 0
            for k in range(N):
                fn,cid = df.iloc[k,1:]
                file_names.append([fn, cid])
                if cid >= 0:
                    self.ind = k
            self.img_list = file_names
        else:
            file_names = self.collect_files(root_dir, file_exten=wildcard)
            file_names = list(set(file_names))
            self.img_list = [[fn,-1] for fn in file_names]
            self.ind = 0
        self.root_dir = root_dir
        self.cur_img_name = None
        self.skip_std = skip_std
        self.show_current_image()

    def collect_files(self, root_dir, file_exten='jpg', files_list=None):
        files_list = [] if files_list is None else files_list
        for dirpath, dirs, files in os.walk(root_dir):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                if fname.endswith(file_exten):
                    files_list.append(fname)
            for dir in dirs:
                cur_dir = os.path.join(dirpath, dir)
                self.collect_files(cur_dir, file_exten=file_exten, files_list=files_list)
        return files_list

    def show_current_image(self, prev_flag=False):
        self.ind = self.ind % len(self.img_list)
        work_done = 100*self.ind/len(self.img_list)
        textstr = f'{self.ind}/{len(self.img_list)},{work_done}'
        print(textstr)
        cid = self.img_list[self.ind][1]
        textstr = f'{self.ind}/{len(self.img_list)},cid:{cid}'
        img_path = self.img_list[self.ind][0]
        self.cur_img_name = img_path
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax_txt.text(0.5, 0.05,textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        img_var = self.skip_std
        img = None
        while img_var <= self.skip_std and not prev_flag:
            img = cv2.imread(img_path)
            img_var = np.var(img)
            print(f'img_var: {img_var}')
            if img_var <= self.skip_std and self.ind < len(self.img_list)-1:
                self.img_list[self.ind][1] = 0
                self.ind += 1
                self.ind = self.ind % len(self.img_list)
                img_path = self.img_list[self.ind][0]
                self.cur_img_name = img_path
            else:
                break
        if img is None:
            img = cv2.imread(img_path)
        textstr = f'{self.ind}/{len(self.img_list)},cid:{cid}'
        if txt_btn is not None:
            txt_btn.label.set_text(textstr)

        ax.imshow(img)
        #plt.title(img_path)
        plt.draw()

    def next(self, event=None):
        self.ind += 1
        self.show_current_image()

    def prev(self, event=None):
        self.ind -= 1
        self.show_current_image(prev_flag=True)

    def ok(self, event):
        print('ok')
        self.img_list[self.ind][1] = 0
        self.next()

    def bad(self, event):
        print('bad')
        self.img_list[self.ind][1] = 1
        self.next()

    def save(self,event):
        print('save')
        user_name = os.getlogin()
        file_name = os.path.join(self.root_dir, f'good_bad_train_{user_name}.csv')
        df = pd.DataFrame(data=self.img_list, columns=["file_name", "class"])
        df.to_csv(file_name)
        print(f'saved file: {file_name}')

root_dir=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set"
print(f'working on directory:{args.root_dir}')

ax_ok = fig.add_axes([0.1, 0.05, 0.1, 0.075])
ax_bad = fig.add_axes([0.21, 0.05, 0.1, 0.075])
ax_save = fig.add_axes([0.32, 0.05, 0.1, 0.075])
axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])

ax_txt = fig.add_axes([0.45, 0.05, 0.2, 0.075])

#ax_txt = fig.add_axes([0.5, 0.05, 0.1, 0.075])
callback = Index(root_dir=args.root_dir, wildcard=args.file_exten, csv_path=args.csv_file, skip_std=args.skip_std)

bok = Button(ax_ok, 'OK')
bok.on_clicked(callback.ok)
bbad = Button(ax_bad, 'BAD')
bbad.on_clicked(callback.bad)
save_b = Button(ax_save, 'Save')
save_b.on_clicked(callback.save)

# Placeholder
txt_btn = Button(ax_txt, '')

bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

#txt_obj = Button(ax_txt, '')
callback.show_current_image()
plt.show()