'''
Created on 2024/06/12
@Author - Dudi Levi
'''

import pytest
from slidecore.predict_imgs.predict_imgs import PredictImgs
import cv2
import os



@pytest.mark.parametrize("img, predictResult, model",
                         [("good_5_11.jpg", "good",
                           "/home/dudi/dev/patology/slidecore/slidecore/model/resnet_epoch_6_acc_92.pt"),])
def test_predictImage(img, predictResult, model):
    pi = PredictImgs(model_path=model)
    imageFile = os.path.join(os.path.dirname(__file__), img)
    cvImg = cv2.imread(imageFile)
    imgs = [cvImg]
    res = pi.predict(imgs)
    print(f'res={res}')

    # im1=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\GoodFocus\GoodFocus_ANONJSBHSI1F2_1_1_level_17_size_840\4_19.jpeg"
    # # im2 doesnt look like a bad focus....
    # im2=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\BadFocus\BadFocus_ANONFACHSI1RE_4_1_level_17_size_840\9_20.jpeg"
    # model_path=r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_6_acc_92.pt"
    # pi = PredictImgs(model_path=model_path)
    # img1 = cv2.imread(im1)
    # img2 = cv2.imread(im2)
    # imgs = [img1, img2]
    # res = pi.predict(imgs)
    # print(f'res={res}')