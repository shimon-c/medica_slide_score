'''
Created on 2024/06/12
@Author - Dudi Levi
'''

import pytest
import cv2
import os
import torch

from slidecore.predict.predict_imgs import PredictImgs
from slidecore.predict.predict_results import GetPredictionResults, GetPredictionClassValue



@pytest.mark.parametrize("img, predictResult, model",
                         [("good_5_11.jpg", "Good",
                           "/home/dudi/dev/patology/slidecore/slidecore/model/resnet_epoch_6_acc_92.pt"),
                           ("bad_96_2.jpg", "NotRelevant",
                           "/home/dudi/dev/patology/slidecore/slidecore/model/resnet_epoch_6_acc_92.pt"),])
def test_predictImage(img, predictResult, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pi = PredictImgs(model_path=model, gpu=-1 if device.type=="cpu" else 0)
    imageFile = os.path.join(os.path.dirname(__file__), img)
    cvImg = cv2.imread(imageFile)
    imgs = [cvImg]
    prediction = pi.predict(imgs)
    print(f'res={prediction}')
    className, res = GetPredictionResults(prediction[0])
    print(f"Target class '{className}' is {res}")
    assert className == predictResult
    res2 = GetPredictionClassValue(prediction[0], className)
    print(f"Target class '{className}' is {res2}")
    assert res == res2

    #predictions = np.array([[0.2, 0.8, 0.1], [0.1, 0.3, 0.6]])  # Sample prediction probabilities
    # predictions = np.array([0.2, 0.8, 0.1])  # Sample prediction probabilities
    # target_class_name = "Good"
    # res = GetPredictionClassResults(predictions, target_class_name)
    # assert res == 0.2
    # print(f"Target class '{target_class_name}' is {res}")
    # target_class_name = "Bad"
    # res = GetPredictionClassResults(predictions, target_class_name)
    # assert res == 0.8
    # print(f"Target class '{target_class_name}' is {res}")
    # target_class_name = "koko"
    # res = GetPredictionClassResults(predictions, target_class_name)
    # assert res == None
    # print(f"Target class '{target_class_name}' is {res} Not exists")

    # predictions = np.array([0.2, 0.8, 0.1])
    # className, res = GetPredictionResults(predictions)
    # assert className == "Bad"
    # assert res == 0.8
    # print(f"Target class '{className}' is {res}")
    # predictions = np.array([0.2, 0.2, 0.6])
    # className, res = GetPredictionResults(predictions)
    # assert className == "NotRelevant"
    # assert res == 0.6
    # print(f"Target class '{className}' is {res}")
    # predictions = np.array([0.2, 0.2, 0.2])
    # className, res = GetPredictionResults(predictions)
    # assert className == None
    # assert res == None
    # print(f"Target class '{className}' is {res}")


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