import torch
import slidecore
import slidecore.net.ensemble
import numpy as np
import cv2
import torchvision


class PredictImgs:
    def __init__(self, model_path=None, gpu=0, ensemble_flag=False):
        if ensemble_flag:
            self.net = slidecore.net.ensemble.Ensemble(model_path=model_path)
        else:
            self.net,args, optim_params,sched_params,epoch = slidecore.net.resnet.ResNet.load(model_path)
        devstr = devstr = f'cuda:{gpu}' if gpu>=0 else 'cpu'
        self.net = self.net.to(devstr)
        self.devstr = devstr
        self.net.eval()
        self.to_tensor = torchvision.transforms.ToTensor()

    @torch.no_grad()
    def predict(self, imgs_list:list=[], inference_flag=False):
        N = len(imgs_list)
        H,W,C = imgs_list[0].shape
        ten = torch.zeros((N,C,H,W), dtype=torch.float32)
        for k in range(N):
            img = imgs_list[k].astype(np.float32)
            img = self.to_tensor(img)
            ten[k,...] = img
        # move to device
        x = ten.to(self.devstr)
        if inference_flag:
            y = self.net.infer(x)
        else:
            y = self.net(x)
        y_np = y.cpu().detach().numpy()
        return y_np

if __name__ == '__main__':
    im1=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\GoodFocus\GoodFocus_ANONJSBHSI1F2_1_1_level_17_size_840\4_19.jpeg"
    # im2 doesnt look like a bad focus....
    im2=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\BadFocus\BadFocus_ANONFACHSI1RE_4_1_level_17_size_840\9_20.jpeg"
    im3=r"C:\Users\shimon.cohen\data\medica\imgdb\db_test_set\test_set\BadFocus\BadFocus_ANONFACHSI1RE_4_1_level_17_size_840\10_22.jpeg"
    model_path=r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_6_acc_92.pt"
    model_path = r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_12_0.921743.pt"
    model_path = r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\model\resnet_epoch_14_acc_93.pt"
    model_path=r"C:\Users\shimon.cohen\PycharmProjects\medica\medica\slidecore\model\resnet_epoch_39_0.892.pt"
    pi = PredictImgs(model_path=model_path)
    img1 = cv2.imread(im1)
    img2 = cv2.imread(im2)
    img3 = cv2.imread(im3)
    imgs = [img1, img2, img3]
    res = pi.predict(imgs)
    print(f'res={res}')

    res = pi.predict(imgs, inference_flag=True)
    print(f'inference_res={res}')
