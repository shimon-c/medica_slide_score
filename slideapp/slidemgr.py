import slidecore.predict.predict_imgs

class SlideMgr:
    def _init__(self,input_dir:str=None, output_dir:str=None, classfier_path:str=None):
        self.predictor = slidecore.predict.predict_imgs.PredictImgs(model_path=classfier_path)
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        pass