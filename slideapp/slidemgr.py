import slidecore.predict.predict_imgs
from slidecore.predict.predict_imgs import PredictImgs as PredictImgs
import os
import utils
import slideapp.config
import shutil
import time
import logging
from datetime import date

#source ~/venv/bin/activate
# source /home/shimon/venv/bin/activate

# Currently need to perform on windows:  pip install albumentations==1.1.0

class SlideMgr:
    def __init__(self,input_dir:str=None, output_dir:str=None, classfier_path:str=None):
        self.predictor = slidecore.predict.predict_imgs.PredictImgs(model_path=classfier_path,
                                                                    cls_tile_thr=slideapp.config.classifer_tile_thr)
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir is not None else input_dir
        self.write_tiles_into_out_dir = slideapp.config.write_tiles_into_out_dir
        self.tiles_working_dir = slideapp.config.tiles_working_dir
        if self.tiles_working_dir != '':
            os.makedirs(self.tiles_working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        log_file = os.path.join(self.output_dir, "slidemgr.log")
        res_file = os.path.join(self.output_dir, "slidemgr_results.txt")
        try:
            os.remove(res_file)
        except Exception as e:
            print(f'Caught: {e}')
        self.res_file = open(res_file, "+w")
        try:
            os.remove(log_file)
        except Exception as e:
            print(f'Failed to remove log msg: {e}')
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG)
        print(f'log_file: {log_file}')

    def run(self):
        prv_date = None
        while True:
            cur_date = date.today()
            print(f'slidemgr date:{cur_date}')
            if prv_date is None or cur_date > prv_date:
                self.work_on_slides(root_dir=self.input_dir,
                                    good_flag=None)
            # Sleep for an hour
            prv_date = cur_date
            time.sleep(60)

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

        # file_names = glob.glob(search_pat, recursive=True)
        file_names = slidecore.predict.predict_imgs.collect_slides(root_dir=root_dir, file_exten=file_exten)
        work_list = []
        num_bad = 0
        num_good = 0
        num_failed = 0
        good_dir, bad_dir = None, None
        if good_flag is None:
            good_dir = os.path.join(self.output_dir, 'good_dir')
            bad_dir = os.path.join(self.output_dir, 'bad_dir')
            os.makedirs(good_dir, exist_ok=True)
            os.makedirs(bad_dir, exist_ok=True)
        for fn in file_names:
            # if a directory was supplied
            dir = self.tiles_working_dir if self.tiles_working_dir != '' else os.path.dirname(fn)
            outputPath = os.path.join(dir, 'tiles')
            logging.info(f'----> Working on slide (tile extractor):{fn}')
            print(f'----> Working on slide (tile extractor):{fn}')
            failed = False
            try:
                extractor = utils.extractor.TileExtractor(slide=fn, outputPath=outputPath, saveTiles=True)
                extractor.run()
                outputPath = extractor.tiles_dir
                out_dir = fn.replace(root_dir, self.output_dir)
                is_bad = pred.predict_from_dir(dir_path=outputPath,
                                               out_dir=out_dir,
                                               percentile = slideapp.config.classifer_slide_thr,
                                               write_tiles_flag=self.write_tiles_into_out_dir)


            except Exception as e:
                logging.error(f'******* Failed on slide:{fn}')
                print(f'******* Failed on slide:{fn}')
                num_failed += 1
                shutil.rmtree(outputPath, ignore_errors=True)
                continue


            shutil.rmtree(outputPath, ignore_errors=True)
            if is_bad:
                num_bad += 1
            else:
                num_good += 1
            work_list.append((out_dir, f'is_bad:{is_bad}'))
            logging.info(f'----->   Slide (after classifier):{fn}, is_bad: {is_bad}')
            sl_res_str = 'Bad' if is_bad else 'Good'
            self.res_file.write(f'{fn}:\t {sl_res_str}\n')
            self.res_file.flush()
            if good_flag is None:
                # Copy the slide for further process
                if is_bad:
                    new_fn = fn.replace(self.input_dir, bad_dir)
                else:
                    new_fn = fn.replace(self.input_dir, good_dir)
                os.makedirs(os.path.dirname(new_fn), exist_ok=True)
                shutil.copy(fn, new_fn)
            # del extractor
        print(f'work_list:\n{work_list}')
        for tp in work_list:
            print(f'{tp}\n')
        num_files = len(file_names) - num_failed
        ret_str = ''
        if good_flag is not None:
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
    ap.add_argument('--run_flag', type=int, default=0, help="Run ")
    args = ap.parse_args()
    return args

if __name__ == "__main__":
    #args = parse_args()
    sm_app = SlideMgr(input_dir=slideapp.config.input_dir,
                      classfier_path=slideapp.config.model_path,
                      output_dir=slideapp.config.out_dir)
    # Check if run mode (not test)
    if slideapp.config.run_flag:
        sm_app.run()
    res_str = f'classifer_tile_thr:{slideapp.config.classifer_slide_thr}\tclassifclassifer_tile_threr_slide_thr:{slideapp.config.classifer_tile_thr}'
    rstr = sm_app.work_on_slides(root_dir=slideapp.config.bad_dir, good_flag=False)
    res_str = f'{res_str}\n{rstr}'
    if os.path.exists(slideapp.config.good_dir):
        rstr = sm_app.work_on_slides(root_dir=slideapp.config.good_dir, good_flag=True)
        res_str = f'{res_str}\n{rstr}'

    print(res_str)
    logging.info(res_str)