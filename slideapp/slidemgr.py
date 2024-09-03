import slidecore.predict.predict_imgs
from slidecore.predict.predict_imgs import PredictImgs as PredictImgs
import os
import utils

class SlideMgr:
    def _init__(self,input_dir:str=None, output_dir:str=None, classfier_path:str=None):
        self.predictor = slidecore.predict.predict_imgs.PredictImgs(model_path=classfier_path)
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir is not None else input_dir

    def run(self):
        pass

    def collect_dirs(self):
        dir_list = []
        for dirpath, dirs, files in os.walk(self.input_dir):
            for dir in dirs:
                dir_list.append(dir)

        return dir_list

    # Work on several slides
    def work_on_slides(self, pred: PredictImgs = None, root_dir: str = None, file_exten='ndpi'):
        search_pat = os.path.join(root_dir, f'**{file_exten}')
        # file_names = glob.glob(search_pat, recursive=True)
        file_names = slidecore.predict.predict_imgs.collect_slides(root_dir=root_dir, file_exten=file_exten)
        work_list = []
        num_bad = 0
        for fn in file_names:
            dir = os.path.dirname(fn)
            outputPath = os.path.join(dir, 'tiles')
            extractor = utils.extractor.TileExtractor(slide=fn, outputPath=outputPath, saveTiles=True)
            extractor.run()
            outputPath = extractor.tiles_dir
            is_bad = pred.predict_from_dir(outputPath)
            num_bad += 1 if is_bad else 0
            work_list.append(outputPath)
            # del extractor
        print(f'work_list:\n{work_list}')


import argparse
def parse_args():
    ap = argparse.ArgumentParser('Ensemble')
    ap.add_argument('--model_path', type=str, required=True, help="Model path where to generate ensmeble")
    ap.add_argument('--inference_size', type=int, default=0, help="perform ineference")
    ap.add_argument('--slides_dir', type=str, default="", help="directory of slides")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    sm_app = SlideMgr(input_dir=args.slides_dir,classfier_path=args.model_path)