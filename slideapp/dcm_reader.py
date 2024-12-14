# https://github.com/pydicom/pydicom
# pip install pydicom
import os

from pydicom import dcmread, pixel_array
from PIL import Image
from pydicom.data import get_testdata_file
import glob
import cv2
# path = r"/home/shimon/Desktop/sectra-9.12.24/bad/ANON6V2ELJ1IK/ANON6V2ELJ1IK_1_4.dcm"
# ds = dcmread(path)
# arr = ds.pixel_array
# arr1 = pixel_array(ds)
# type(ds.PixelData)

class DicomExtractor:
    def __init__(self, file_path=None, outputPath=None):
        ds = dcmread(file_path)
        self.ds = ds
        self.data = pixel_array(ds)
        self.out_path = outputPath
        self.tiles_dir = None

    def run(self):
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


