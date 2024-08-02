'''
Created on Mar 5, 2024

@author: dudi
'''

#from multiprocessing import
import os
import sys
import glob
import pathlib
import torch
import time
from torch.multiprocessing import Process, set_start_method, JoinableQueue, Queue
from fastcore.all import *
from fastai.vision.core import PILImage
from fastai.learner import load_learner
import argparse
import logging
from slidems.common.logi import CreateLogger, Duration
from slidems.common.metadata_io import Metadata

log = logging.getLogger(__name__)

class TilesCheckerWorker(Process):

    def __init__(self, workerId, workQueue, resultsQueue, learn,
                 saveClassifiedTiles, outputPath, imgFormat="jpg"):
        Process.__init__(self, name='TilesCheckerWorker')
        self.daemon = True
        self._workerId = workerId
        self._queue = workQueue
        self._model = learn
        self._results = resultsQueue
        self._saveClassifiedTiles = saveClassifiedTiles
        self._imgFormat = imgFormat
        self._outputPath = outputPath
        self._workerSumResults = {"GoodFocus":0, "BadFocus": 0, "NotRelevant": 0}
        programName = os.path.basename(__file__).split('.')[0]
        #self._log = CreateLogger(name=f"{programName}_Worker{self._workerId}",
        #                         logFile=os.path.join(self._outputPath,
        #                        f"{programName}_Worker{self._workerId}.log"))
    @Duration(log, msg="Preparing device", logStart=False)
    def _prepare(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._learn = load_learner(fname=self._model, cpu=device.type=="cpu")
        self._learn.to(device)
        log.debug(f"TilesCheckerWorker {self._workerId} Loaded model - {self._model} to device {device}")
        totalTime = time.perf_counter() - self.startTime
        log.debug(f"END: Prepare TilesCheckerWorker {self._workerId} (Time: {totalTime:.3f} Sec)")

    def run(self):
        self.startTime = time.perf_counter()
        self._prepare()
        log.debug(f"Start: TilesCheckerWorker {self._workerId}")
        run = True
        while run:
            data = self._queue.get()
            if data is None:
                break
            run = self._doPredict(data)

        self._results.put(self._workerSumResults)
        log.info(f"Worker {self._workerId} {self._workerSumResults}")
        totalTime = time.perf_counter() - self.startTime
        log.debug(f"END: TilesCheckerWorker {self._workerId} (Time: {totalTime:.3f} Sec)")
        self._queue.task_done()

    @Duration(log, msg="doPredict", logStart=False)
    def _doPredict(self, data):
        tilePath, tileName, tile, outputPath = data
        if tile:
            img = tile
        else:
            img = PILImage.create(os.path.join(tilePath, tileName))

        className, classNum, probs = self._predict(img)
        self._workerSumResults[className] += 1
        #log.debug(f"Tile {tileIndex} is a: {className}. with probability {probs[classNum]:.4f}", end='\r')
        if self._saveClassifiedTiles:
            self._save_tile(img, os.path.join(f"{outputPath}_{className}", tileName))
        self._queue.task_done()
        return True

    #@Duration(log, msg="predict", logStart=False)
    def _predict(self, img):
        return self._learn.predict(img)

    def _save_tile(self, img, tilePath):
        if not os.path.exists(tilePath):
            img.save(tilePath, quality=90, icc_profile=img.info.get('icc_profile'))
        else:
            log.debug(f"File exists  - {tilePath}")



class CheckTilesFocus():
    """Evaluate all slide images and determine if the slide is good"""

    def __init__(self, tilesInputDir=None, saveResults=None,
                 workers=2, outputPath="output", checkTilesQueud=None,
                 slide=None, imgFormat="jpg"):
        self._workers = workers
        self._sourceSlideFile = slide
        self._tilesInputDir = tilesInputDir
        self._saveClassifiedTiles = saveResults
        self._imgFormat = imgFormat
        if checkTilesQueud is None:
            self._checkTilesQueud = JoinableQueue(2 * self._workers)
        else:
            self._checkTilesQueud = checkTilesQueud
        self._resultQueue = Queue(self._workers+2)
        self._results = {"GoodFocus":0, "BadFocus": 0, "NotRelevant": 0}
        self._processed = 0
        self._outputPath = outputPath

        modeFileName = "resize512_RandomResizedCrop256_resnet50_epoc10.pkl"
        model = os.path.join(os.path.join(Path(__file__).parent.parent, "configs", "model", modeFileName))
        for _i in range(self._workers):
            TilesCheckerWorker(workerId=_i+1,
                                workQueue=self._checkTilesQueud,
                                resultsQueue=self._resultQueue,
                                learn=model,
                                saveClassifiedTiles=self._saveClassifiedTiles,
                                outputPath=self._outputPath
                                ).start()

    @Duration(log, msg="CheckTilesFocus.run", logStart=True)
    def run(self):
        if self._tilesInputDir:
            # work with tiles from files
            assert os.path.isdir(self._tilesInputDir)
            log.info(f"Tiles Input Directory: {self._tilesInputDir}")
            tilesOutputDir = os.path.join(self._outputPath, os.path.basename(self._tilesInputDir))
            if self._saveClassifiedTiles:
                log.info(f"Save Result Classified Tiles: {self._saveClassifiedTiles}")
                for key in self._results.keys():
                    pathlib.Path(f"{tilesOutputDir}_{key}").mkdir(parents=True, exist_ok=True)
            #tiles = os.listdir(self._tilesInputDir)
            pattern = os.path.join(self._tilesInputDir, f"*.{self._imgFormat}")
            tiles = glob.glob(pattern)
            self.totalTiles = len(tiles)
            log.debug(f"Total tiles to process: {self.totalTiles}")
            for i, tile in enumerate(tiles):
                tileName = os.path.basename(tile)
                self._checkTilesQueud.put((self._tilesInputDir, tileName, None, tilesOutputDir))
                self._processed += 1
                #self.printTileDone()
            self.shutdown()
        else:
            log.warning("CheckTilesFocus expect to receive PIL image data")



    def _sumWorkersResults(self, path=None):
        while not self._resultQueue.empty():
            res = self._resultQueue.get()
            for key in res:
                self._results[key] = self._results[key] + res[key]

        maxTiles = 0
        maxClassName = ""
        for key in self._results:
            if self._results[key] > maxTiles:
                maxTiles = self._results[key]
                maxClassName = key

        if path:
            tilesInputDir = path
        else:
            tilesInputDir = self._tilesInputDir
        # with open(f"{tilesInputDir}/meta_data.txt", "a") as mFile:
        #     mFile.write(f"\nSlide - {self._sourceSlideFile}\n")
        #     mFile.write(f"Tiles input dir - {tilesInputDir}\n")
        #     mFile.write(f"Total results = {self._results}\n")
        #     mFile.write(f"\nSlide classified as - {maxClassName}\n")

        meta = Metadata()
        meta.slideName = self._sourceSlideFile
        meta.tilesInputDir = tilesInputDir
        meta.evaluationResults = self._results
        meta.slideClassification = maxClassName
        meta.save(file=f"{tilesInputDir}/metadata.json")

        log.info(f"Tiles - {tilesInputDir}")
        log.info(f"Total results = {self._results}")
        log.info(f"Slide classified as - {maxClassName}")
        #os.rename(f"{tilesInputDir}/meta_data.txt", f"{tilesInputDir}/meta_data_{maxClassName}.txt")

    def printTileDone(self):
        count, total = self._processed, self.totalTiles
        if count % 100 == 0 or count == total:
            print(" " * (25 - 1), end="") # move cursor to 25 position
            print(f"\nEvaluated : {count}/{total} tiles", end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def shutdown(self, path=None):
        log.debug("Start CheckTilesFocus shutdown")
        for _i in range(self._workers):
            self._checkTilesQueud.put(None)
        self._checkTilesQueud.join()
        log.debug("After CheckTilesFocus shutdown join")
        self._sumWorkersResults(path)


def TilesChecker_args():
    parser = argparse.ArgumentParser(description='Tiles Checker')
    parser.add_argument('-i','--input_tiles', type=str,
                        default=os.path.join(Path(__file__).parent.parent.parent, "output/tiles/2024-01-15_21.26.19_level_17_size_840"),
                        help="Full path to tiles directory. [default: %(default)s]")
    parser.add_argument('-o','--output_path', type=str,
                        default=os.path.join(Path(__file__).parent.parent.parent, "output", "tiles"),
                        help="Output image path. [default: %(default)s]")
    parser.add_argument('--format', metavar='{jpg|png}',
                        default="jpg",
                        help="Output image format. [default: %(default)s]")
    parser.add_argument('--save_results', metavar='{True|False}',
                        default=True,
                        help="Save classified tiles. [default: %(default)s]")

    args, unknown = parser.parse_known_args()
    return args



if __name__ == '__main__':
    print(sys.argv)
    args = TilesChecker_args()
    programName = os.path.basename(__file__).split('.')[0]
    CreateLogger(logFile=os.path.join(args.output_path, f"{programName}.log"))
    if torch.cuda.is_available():
        log.debug("GPU is available!")
        try:
            set_start_method('spawn')
        except RuntimeError:
            log.warn("set_start_method raised RuntimeError")
            pass
    else:
        log.warn("GPU not detected. Training will run on CPU.")

    workers = 1
    evaluateQueue = JoinableQueue(2 * workers)
    checker = CheckTilesFocus(tilesInputDir=args.input_tiles,
                              outputPath=args.output_path,
                              workers=workers,
                              saveResults=args.save_results,
                              checkTilesQueud=evaluateQueue)
    checker.run()



    # CheckTilesFocus(tilesInputDir="/home/dudi/privet/med/biopsyfocus/output/tiles/2024-01-15_21.26.19_level_17_size_840").run()
    # CheckTilesFocus(tilesInputDir="/home/dudi/privet/med/openslide/output_images/0305/ANONKMHBUI1RK_1_1_level_17_size_840_INF").run()
    # CheckTilesFocus(tilesInputDir="/home/dudi/privet/med/openslide/output_images/0305/ANONKMHBUI1RK_2_1_level_17_size_840_OOF").run()
    # CheckTilesFocus(tilesInputDir="/home/dudi/privet/med/openslide/output_images/0305/ANONS0IBUI175_1_1_level_17_size_840_INF").run()
    # CheckTilesFocus(tilesInputDir="/home/dudi/privet/med/openslide/output_images/0305/ANONB5IBUI1M9_1_1_level_17_size_840_INF").run()
    # CheckTilesFocus(tilesInputDir="/home/dudi/privet/med/openslide/output_images/0305/ANONF2IBUI1F2_1_1_level_17_size_840_OOF").run()
    # CheckTilesFocus(tilesInputDir="/home/dudi/privet/med/openslide/output_images/0305/ANONF2IBUI1F2_2_1_level_17_size_840_OOF").run()

    print("Done!")

