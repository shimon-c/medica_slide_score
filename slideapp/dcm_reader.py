# https://github.com/pydicom/pydicom
# pip install pydicom
import os
import pydicom
from pydicom import dcmread, pixel_array
from PIL import Image
from pydicom.data import get_testdata_file
import glob
import cv2
import pathlib
# path = r"/home/shimon/Desktop/sectra-9.12.24/bad/ANON6V2ELJ1IK/ANON6V2ELJ1IK_1_4.dcm"
# ds = dcmread(path)
# arr = ds.pixel_array
# arr1 = pixel_array(ds)
# type(ds.PixelData)

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

dicom_path = 'path/to/tile1.dcm'

def read_sitk(imgs_list):
    images = []
    k  = 0
    for img_p in imgs_list:
        k += 1
        if k>=3:
            break
        try:
            image = sitk.ReadImage(img_p)
            array = sitk.GetArrayFromImage(image)
            print("Successfully accessed pixel array with SimpleITK. Shape:", array.shape)
            images.append(array)
            plt.imshow(array[0], cmap='gray')
            plt.show()
        except Exception as e:
            print(f"Error accessing pixel array with SimpleITK: {e}, {img_p}")
    stitched_image = cv2.vconcat([cv2.hconcat(images[:2]), cv2.hconcat(images[2:4])])
    plt.imshow(stitched_image, cmap='gray')
    plt.show()


# Function to read DICOM tiles and stitch them together
def stitch_tiles(tile_paths, output_path):
    images = []
    for tile_path in tile_paths:
        try:
            ds = pydicom.dcmread(tile_path)
            image = ds.pixel_array
            images.append(image)
        except Exception as e:
            print(f'Error reading: {tile_path}: {e}')

    # Combine images into a single slide
    stitched_image = cv2.vconcat([cv2.hconcat(images[:2]), cv2.hconcat(images[2:4])])

    # Save the stitched image
    cv2.imwrite(output_path, stitched_image)
    print(f"Stitched image saved to {output_path}")

# Paths to your DICOM tiles
def get_dicoms_files(dir):
    dir_name = os.path.join(dir,"*dcm")
    file_names = glob.glob(dir_name)
    return file_names

# Stitch the tiles
test_stitch = False
if test_stitch:
    tile_paths = get_dicoms_files('/home/shimon/Desktop/sectra-9.12.24/bad/ANON6V2ELJ1IK')
    output_path = 'path/to/save/your/slide_image.png'
    read_sitk(tile_paths)
    stitch_tiles(tile_paths, output_path)

class DicomExtractor:
    def __init__(self, file_path=None, outputPath=None):
        self.file_path = file_path
        self.out_path = outputPath
        self.tiles_dir = None

    def run(self):
        path = pathlib.Path(self.file_path )

        if  path.is_symlink():
            self.tiles_dir = None
            return
        ds = dcmread(self.file_path)
        self.ds = ds
        self.data = pixel_array(ds)

        out_dir = os.path.join(self.out_path, 'tiles')
        self.tiles_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        if len(self.data.shape) ==3:
            H,W,C = self.data.shape
            self.data = self.data.reshape((1,H,W,C))
        N = self.data.shape[0]
        for k in range(N):
            file = os.path.join(out_dir, f't_{k}.jpg')
            #im = Image.fromarray(self.data[k,...])
            #im.save(file + ".jpg", "JPEG")
            cv2.imwrite(file, self.data[k,...])


