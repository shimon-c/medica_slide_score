# https://realpython.com/pysimplegui-python/

# python -m pip install pysimplegui
""" License
e9yNJXMlayWkNcl2bUn1NAlBVhHKl9wJZuSEID6VIMkNR3lZdrmeV9s7bF3cBxl1cYiGIOsOI2kNxYp9YQ2ZVquFc22vVPJRRNChII6kMDTucXzyMpT8Ab1KOBTnYg2hOrSSwCiLTyGzl4jBZoWw5TzlZrUaRgl2cXGyxEv0egWP1SlCbSn5RuWBZgXGJ8zuaaWa9iuFIujUoNxzLOC7JgOuYCWu1aleRzmglnyXce3EQ7ikOSiqJczma8GQl4trbJ2X4uiMLQCrJqOxYlWf11liT5GOFDzJd9ChII6RIVmvNuvna5G4VyumIqi1wHidQm2Z9qtlcfGdF3uXeZShI76UITiHI8szIGkcNo1ecz3KROvRbmWgVzypSAUkQHiZOeiCIA02NnTxY7yoNPCLIQsXInktR0hJdAGdVlJAcj3aNO1aZDWdQdiGOciyI2xfMuS98hw4OSCD8UyPMpDrI402IxiPwOiORiG3Fz0sZAUbVz42cPG3luycZBXgMMikOii6ISx4MhSO8GwhOnCx8lyjMgDaIG1LIOiEwUiiRuWT1LhrajWCx2BEZJG2RJyZZWX3NezDI3juoGiBca2NhlpFbUWd9auXLpm0NtvnaTGrVpuXMETck01VOpExBunqbZWuFHpnb8Cc5vj4bJ2100iELoCIJsJ7UDESF2k6ZEHOJTlicM3bMeibOyivIJ3FNayn4PxMMSjkU8uBMUTrgg1aLajdYF0fIynP0d=l3ab536f0e3ccddae6154976ed946974d14fcfc4580f5adb9bdb8b935bd028c066a9f4288f96bb5cf1a2b01fcc26c904fcaa23ea50c47a1c568684a59a3a993e95576114b2d8c25fdcc3f853d5b43dd8fa6e9b83fdb04b5405a74ff8409f99bf994815ebff6921323ecb0cb8d7530142cd22a99a3b4afe1fb0999f1da246d24b35710ccc348b5e78458750eab34a180661a73ad6b2314d867c718a967e6dfd1977eda01c6b8d3be922206febcefc982d9cafd6e1fb48ae2a98ace5c0bbe4c1db86ac748030b9154b63d78b45381571da4c6aa8df77bd58402f3208ab5eaa3ae5fe1b650245935f3a54b83411bd6ee8eb04f36117aa928e63a0f5f30ab2fab0914f447e6d1f92d8c559ffdf490e7bbbef0ef5c83a7cbad3e04ed52238228e12667090bf766930db958784e6189f76f16f2828a965db4d7360bfb4e4da14d6c2392fab87fe9495122c0938f4d421acb838cd6cabd8bce66f442ad7e5974dab6c843f620ec3f4bb92a60bad2edb0f868e2033a6d24b522e064dbff32047680e47710f01a084caad2227a022609d6448b7ba07814e42acb183356f09c8cbd522336b10ff7efda800af3bff5e9e546492dbd11265b53548635170ba902fb30367fd8855152fd86e13583a86e074785fa99f5819109fefc35d1a00d6d737b369048b903cbecc34461c45042c735a5e095203529293032b4c4bb19de82583412985f3ba8
"""
import sys

import PySimpleGUI as sg

#sg.Window(title="Hello World", layout=[[]], margins=(100, 50)).read()

# img_viewer.py

import PySimpleGUI as sg
import os.path
import cv2
import tkinter
# First the window layout in 2 columns

import tkinter as tk
from tkinter import *
from PIL import ImageTk,Image

root = tk.Tk()
root.title("Tiles GUI")
root.geometry("400x200")
img1_path=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg"
img = Image.open(img1_path)
#image_1 = tk.PhotoImage(img).subsample(2, 2)
cv_img = cv2.imread(img1_path)
img = Image.fromarray(cv_img)
image_1 = tk.PhotoImage(img)
image_2 = None #tk.PhotoImage(file="geeksforgeeks-logo.png")
#label_pic = tk.Label(root)
canvas = Canvas(root, width=650, height= 350)
image_container =canvas.create_image(0,0, anchor="nw",image=image_1)
# Function to update the image in label
def update_image_in_canvas(path=""):
    global img
    global image_2
    img = Image.open(path)
    img = ImageTk.PhotoImage(Image.open(path))
    image_2 = tk.PhotoImage(img)
    canvas.itemconfig(image_container, image=image_2)


def good_act():
    print('Good act')

def bad_act():
    print('Bad act1')

def next_act():
    print('Next act')
    update_image_in_canvas(path=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_4_11.jpeg")

# Adding a button
good_button = tk.Button(root, text="Good", command=good_act)
good_button.pack()
bad_button = tk.Button(root, text="Bad", command=bad_act)
bad_button.pack()
next_button = tk.Button(root, text="Next", command=next_act)
next_button.pack()
canvas.pack()
root.mainloop()

class CanvasObj:
    def __init__(self, root, canvas=None, height=300, width=500):
        self.root = root
        self.canvas = Canvas(root, height=300, width=500) if canvas is None else canvas
        self.canvas.pack()
        self. img = None

    def create_image(self,img_path):
        print('CanvasObj create img')
        self.img = ImageTk.PhotoImage(Image.open(img_path))
        print(f'image read: {img_path}')
        self.canvas.create_image(400, 400, anchor=NW, image=self.img)
        self.canvas.pack()

canvas = Canvas(root,height=300, width=500)
img = ImageTk.PhotoImage(Image.open(r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg"))
canvas.create_image(400,200, anchor=NW, image=img)
canvas.pack()
root.mainloop()

sys.exit(0)
# canvas = CanvasObj(root=root, canvas=canvas)
# canvas.create_image(img_path=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg")
def create_tk_img(img_path):
    img = ImageTk.PhotoImage(Image.open(img_path))
    print(f'image read: {img_path}')
    canvas.create_image(400, 400, anchor=NW, image=img)
    #canvas.pack()

#img = PhotoImage(file=r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg")
#img = Image.open(r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg")
#img = ImageTk.PhotoImage(Image.open(r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg"))
#canvas.create_image(400,200, anchor=NW, image=img)
#canvas.create_tk_img(r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_1_9.jpeg")
#canvas.pack()

# Adding a label
#label = tk.Label(root, text="Hello, Tkinter!")
#label.pack()

# def good_act():
#     print('Good act')
#
# def bad_act():
#     print('Bad act1')
#
# def next_act():
#     print('Next act')
#     img = ImageTk.PhotoImage(Image.open(
#         r"C:\Users\shimon.cohen\data\medica\imgdb\db_train_set\train_set\BadFocus\ANONFACHSI1RE_2_1_4_11.jpeg"))
#     canvas.create_image(400, 200, anchor=NW, image=img)
#     canvas.pack()

# Adding a button
good_button = tk.Button(root, text="Good", command=good_act)
good_button.pack()
bad_button = tk.Button(root, text="Bad", command=bad_act)
bad_button.pack()
next_button = tk.Button(root, text="Next", command=next_act)
next_button.pack()


# Start the application
root.mainloop()

sys.exit(0)
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    #[sg.Image(key="-IMAGE-")],
    [sg.Image(filename="", key="-IMAGE-")],
]
button_list =[
    [sg.Button("OK"), sg.Button("Bad"),sg.Button("Next"),sg.Button("Prev")],
]
# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
        sg.HorizontalSeparator(),
        sg.HorizontalLine(button_list),
    ]
]

window = sg.Window("Image Viewer", layout)

# Events loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", "jpeg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            img = cv2.imread(filename)
            cv2.imshow(img)
            print(f'read image:{filename} shape: {img.shape}')
            window["-IMAGE-"].update(data=img)
        except:
            pass

window.close()