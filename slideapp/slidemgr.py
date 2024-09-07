import slidecore.predict.predict_imgs
from slidecore.predict.predict_imgs import PredictImgs as PredictImgs
import os
import utils
import slideapp.config
import shutil

class SlideMgr:
    def __init__(self,input_dir:str=None, output_dir:str=None, classfier_path:str=None):
        self.predictor = slidecore.predict.predict_imgs.PredictImgs(model_path=classfier_path,
                                                                    cls_tile_thr=slideapp.config.classifer_tile_thr)
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir is not None else input_dir
        self.write_tiles_into_out_dir = slideapp.config.write_tiles_into_out_dir

    def run(self):
        pass

    def collect_dirs(self, input_dir = None):
        dir_list = []
        input_dir = input_dir if input_dir is not None else self.input_dir
        for dirpath, dirs, files in os.walk(self.input_dir):
            for dir in dirs:
                dir_list.append(dir)

        return dir_list

    # Work on several slides
    def work_on_slides(self, root_dir: str = None, file_exten='ndpi',good_flag=False):
        pred = self.predictor
        search_pat = os.path.join(root_dir, f'**{file_exten}')
        # file_names = glob.glob(search_pat, recursive=True)
        file_names = slidecore.predict.predict_imgs.collect_slides(root_dir=root_dir, file_exten=file_exten)
        work_list = []
        num_bad = 0
        num_good = 0
        for fn in file_names:
            dir = os.path.dirname(fn)
            outputPath = os.path.join(dir, 'tiles')
            extractor = utils.extractor.TileExtractor(slide=fn, outputPath=outputPath, saveTiles=True)
            extractor.run()
            outputPath = extractor.tiles_dir
            out_dir = fn.replace(root_dir, self.output_dir)
            is_bad = pred.predict_from_dir(dir_path=outputPath,
                                           out_dir=out_dir,
                                           percentile = slideapp.config.classifer_slide_thr,
                                           write_tiles_flag=self.write_tiles_into_out_dir)
            shutil.rmtree(outputPath, ignore_errors=True)
            if is_bad:
                num_bad += 1
            else:
                num_good += 1
            work_list.append((out_dir, f'is_bad:{is_bad}'))
            # del extractor
        print(f'work_list:\n{work_list}')
        for tp in work_list:
            print(f'{tp}\n')
        num_files = len(file_names)
        ret_str = ''
        if good_flag:
            FB = num_bad / num_files
            ret_str = f'Good scan False bad:{FB}, scanned:{num_files}'
            print(ret_str)
        else:
            TB = num_bad / num_files
            ret_str = f'Bad Scan True Bad:{TB}, scanned:{num_files}'
            print(ret_str)
        return ret_str


import argparse
def parse_args():
    ap = argparse.ArgumentParser('Ensemble')
    ap.add_argument('--model_path', type=str, required=True, help="Model path where to generate ensmeble")
    ap.add_argument('--inference_size', type=int, default=0, help="perform ineference")
    ap.add_argument('--slides_dir', type=str, default="", help="directory of slides")
    ap.add_argument('--out_dir', type=str, default="", help="directory of slides")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    #args = parse_args()
    sm_app = SlideMgr(input_dir=slideapp.config.bad_dir,
                      classfier_path=slideapp.config.model_path,
                      output_dir=slideapp.config.out_dir)
    res_str = sm_app.work_on_slides(root_dir=slideapp.config.bad_dir, good_flag=False)

    if os.path.exists(slideapp.config.good_dir):
        rstr = sm_app.work_on_slides(root_dir=slideapp.config.good_dir, good_flag=True)
        res_str = f'{res_str}\n{rstr}'
    print(res_str)