import sys

import slidecore.predict.predict_imgs
from slidecore.predict.predict_imgs import PredictImgs as PredictImgs
import os
import utils
import slideapp.config
import shutil
import time
import logging
from datetime import date
import cv2
import datetime
import slideapp.slide_access_time
import slideapp.dcm_reader
# https://www.geeksforgeeks.org/send-mail-attachment-gmail-account-using-python/
import smtplib          # to send emails every day

#working dir: /mnt/medica/medica_lab_project/medica_slide_score/
#source ~/venv/bin/activate
# source /home/shimon/venv/bin/activate

# Currently need to perform on windows:  pip install albumentations==1.1.0
# run command: /mnt/medica/medica_lab_project/medica_slide_score$ python ./slideapp/slidemgr.py
SEC = 60
HOUR = SEC * 60
DAY = HOUR * 24

class SlideMgr:
    LAST_RUN_FNAME = 'last_run.txt'
    def __init__(self,input_dir:str=None, output_dir:str=None, classfier_path:str=None):
        self.predictor = slidecore.predict.predict_imgs.PredictImgs(model_path=classfier_path,
                                                                    cls_tile_thr=slideapp.config.classifer_tile_thr)
        self.input_dir = input_dir
        self.output_dir = output_dir if output_dir is not None else input_dir
        self.write_tiles_into_out_dir = slideapp.config.write_tiles_into_out_dir
        self.tiles_working_dir = slideapp.config.tiles_working_dir
        self.slide_ds_path = os.path.join(slideapp.config.out_dir, 'downsample_imgs') if slideapp.config.downsample_slide>0 else None
        if self.tiles_working_dir != '':
            os.makedirs(self.tiles_working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.last_run = slideapp.slide_access_time.SlideAccessTime()
        last_run_name = os.path.join(self.output_dir, SlideMgr.LAST_RUN_FNAME)
        if os.path.exists(last_run_name):
            self.last_run.set_last_time_from_file(filename=last_run_name)
        log_file = os.path.join(self.output_dir, "slidemgr.log")
        res_file = os.path.join(self.output_dir, "slidemgr_results.txt")
        try:
            os.remove(res_file)
        except Exception as e:
            print(f'Caught: {e}')
        self.res_file_str = res_file
        self.res_file = open(self.res_file_str, "+w")
        try:
            os.remove(log_file)
        except Exception as e:
            print(f'Failed to remove log msg: {e}')
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG)
        print(f'log_file: {log_file}')

    def run(self, max_iters=7):
        prv_date = None
        iter = 0
        while True and iter < max_iters:
            cur_date = date.today()
            current_time = datetime.datetime.now()
            hour = current_time.hour
            print(f'slidemgr date:{hour}, {current_time.day}/{current_time.month}/{current_time.year} \n')
            if prv_date is None or cur_date > prv_date:
                self.work_on_slides(root_dir=self.input_dir,
                                    file_exten=slideapp.config.input_file_exten,
                                    good_flag=None)
            # Sleep for an hour
            prv_date = cur_date
            time.sleep(DAY)
            iter += 1

    def collect_dirs(self, input_dir = None):
        dir_list = []
        input_dir = input_dir if input_dir is not None else self.input_dir
        for dirpath, dirs, files in os.walk(self.input_dir):
            for dir in dirs:
                dir_list.append(dir)

        return dir_list
    def filter_files(self, files=[], filter_str = '01'):
        ret_files = []
        for fn in files:
            bfn = os.path.basename(fn)
            filtered_list = bfn.split('-')
            dcm_flag = bfn.endswith('dcm')
            if len(filtered_list)>3 or dcm_flag:
                if dcm_flag or filter_str in filtered_list:
                    ret_files.append(fn)
        return ret_files

    # Work on several slides
    def work_on_slides(self, root_dir: str = None, file_exten='ndpi',good_flag=False):
        pred = self.predictor
        cur_run = slideapp.slide_access_time.SlideAccessTime()
        cur_run.set_current_time()
        # update last run
        last_run_name = os.path.join(self.output_dir, SlideMgr.LAST_RUN_FNAME)
        #cur_run.save_current_time(filename=last_run_name)
        # file_names = glob.glob(search_pat, recursive=True)
        file_names = slidecore.predict.predict_imgs.collect_slides(root_dir=root_dir, file_exten=file_exten)
        # Just for now filter colored slices and those which were already scanned
        file_names = self.filter_files(files=file_names)
        file_names = self.last_run.filter_files(files_list=file_names)
        file_names = list(set(file_names))
        work_list = []
        num_bad = 0
        num_good = 0
        num_failed = 0
        good_dir, bad_dir = None, None
        print(f'\n---> work_on_slides: collected {len(file_names)} slides  <---\n')
        if good_flag is None:
            shutil.rmtree(self.output_dir, ignore_errors=True)
            os.makedirs(self.output_dir, exist_ok=True)
            if self.slide_ds_path is not None:
                os.makedirs(self.slide_ds_path, exist_ok=True)
            self.res_file = open(self.res_file_str, "+w")
            good_dir = os.path.join(self.output_dir, 'good_dir')
            bad_dir = os.path.join(self.output_dir, 'bad_dir')
            os.makedirs(good_dir, exist_ok=True)
            os.makedirs(bad_dir, exist_ok=True)
            cur_run.save_current_time(filename=last_run_name)

        for kfn,fn in enumerate(file_names):
            # if a directory was supplied
            dir = self.tiles_working_dir if self.tiles_working_dir != '' else os.path.dirname(fn)
            outputPath = os.path.join(dir, 'tiles')
            logging.info(f'----> Working on slide (tile extractor):{fn}')
            print(f'----> Working on slide (tile extractor):{fn}')
            failed = False
            ds_img = None
            try:
                if file_exten != 'dcm':
                    extractor = utils.extractor.TileExtractor(slide=fn, outputPath=outputPath,
                                                              saveTiles=True, std_filter=0)
                else:
                    extractor = slideapp.dcm_reader.DicomExtractor(file_path=fn, outputPath=outputPath)
                extractor.run()
                outputPath = extractor.tiles_dir
                base_fn = os.path.basename(fn)
                out_dir = os.path.join(self.output_dir, base_fn)
                is_bad,slide_img,ds_img = pred.predict_from_dir(dir_path=outputPath,
                                               out_dir=out_dir,
                                               percentile = slideapp.config.classifer_slide_thr,
                                               write_tiles_flag=self.write_tiles_into_out_dir,
                                               tiles_list=extractor.tiles_list,
                                               tile_w=extractor.tile_size, tile_h=extractor.tile_size,
                                               n_tile_rows=extractor.rows, n_tile_cols=extractor.cols)


            except Exception as e:
                fn = file_names[kfn]
                logging.error(f'******* Failed on slide:{fn}')
                print(f'******* Failed on slide:{fn}')
                self.res_file.write(f'******* Failed on slide:{fn}\n')
                self.res_file.flush()
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
            basename = os.path.basename(fn)
            # Use to be the full path
            self.res_file.write(f'{basename},\t {sl_res_str}\n')
            self.res_file.flush()
            if good_flag is None:
                # Copy the slide for further process
                basename = os.path.basename(fn)
                if is_bad:
                    #new_fn = fn.replace(self.input_dir, bad_dir)
                    new_fn = os.path.join(bad_dir, basename)
                else:
                    new_fn = os.path.join(good_dir, basename)
                    #new_fn = fn.replace(self.input_dir, good_dir)
                try:
                    # Catch system errors
                    os.makedirs(os.path.dirname(new_fn), exist_ok=True)
                    shutil.copy(fn, new_fn)
                    new_fn = f'{new_fn}.jpg'
                    if slide_img is not None:
                        cv2.imwrite(filename=new_fn,img=slide_img)
                except Exception as e:
                    print(f'caught exception: {e}')
            # del extractor
        print(f'work_list:\n{work_list}')
        for tp in work_list:
            print(f'{tp}\n')
        print(f'num_failed:{num_failed}, num files: {len(file_names)}, num_good:{num_good}, num_bad:{num_bad}')
        num_files = len(file_names) - num_failed
        if num_files<=0: num_files=1
        ret_str = ''
        num_bad_prob = num_bad/num_files
        num_good_prob = num_good/num_files
        print(f'bad%:{num_bad_prob}, good%:{num_good_prob}')
        if good_flag is not None:
            if good_flag:
                FB = num_bad / num_files
                ret_str = f'Good scan False bad:{FB}, scanned:{num_files}'
                print(ret_str)
            else:
                TB = num_bad / num_files
                ret_str = f'Bad Scan True Bad:{TB}, scanned:{num_files}'
                print(ret_str)
        print(f'results:{self.res_file_str}')
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
        sys.exit(0)
    res_str = f'classifer_tile_thr:{slideapp.config.classifer_slide_thr}\tclassifclassifer_tile_threr_slide_thr:{slideapp.config.classifer_tile_thr}'
    rstr = sm_app.work_on_slides(root_dir=slideapp.config.bad_dir, good_flag=False)
    res_str = f'{res_str}\n{rstr}'
    if os.path.exists(slideapp.config.good_dir):
        rstr = sm_app.work_on_slides(root_dir=slideapp.config.good_dir, good_flag=True)
        res_str = f'{res_str}\n{rstr}'

    print(res_str)
    logging.info(res_str)