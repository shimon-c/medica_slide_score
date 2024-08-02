'''
Created on 2024/06/18 12:20

@Author - Dudi Levi
'''

import logging
from torch.multiprocessing import Process, set_start_method, JoinableQueue, Queue
import time
import os

from slidems.common.logi import Duration
from slidecore.predict.predict_imgs import PredictImgs


log = logging.getLogger(__name__)


class TilesClassifierWorker(Process):
    """ Classify the Recive streem of tiles/images """

    def __init__(self, workerId, workQueue, resultsQueue, args):
        Process.__init__(self, name='TilesClassifierWorker')
        self.daemon = True
        self._workerId = workerId
        self._queue = workQueue
        self._results = resultsQueue

        self._model = args["model"]

        self._saveClassifiedTiles = args["saveClassifiedTiles"]
        self._imgFormat = args["imgFormat"]
        self._outputPath = args["outputPath"]
        self._workerSumResults = {cl:0 for cl in args["classes"]}

    @Duration(log, msg="Preparing device", logStart=False)
    def _prepare(self):
        self.startTime = time.perf_counter()
        self._pi = PredictImgs(model_path=self._model)

        log.debug(f"TilesClassifierWorker {self._workerId} Loaded model - {self._model}")
        totalTime = time.perf_counter() - self.startTime
        log.debug(f"END: Prepare TilesClassifierWorker {self._workerId} (Time: {totalTime:.3f} Sec)")

    def run(self):
        self._prepare()

        log.debug(f"Start: TilesClassifierWorker {self._workerId}")
        run = True
        while run:
            data = self._queue.get()
            if data is None:
                break
            run = self._doPredict(data)

        self._results.put(self._workerSumResults)
        log.info(f"Worker {self._workerId} {self._workerSumResults}")
        totalTime = time.perf_counter() - self.startTime
        log.debug(f"END: TilesClassifierWorker {self._workerId} (Time: {totalTime:.3f} Sec)")
        self._queue.task_done()

    @Duration(log, msg="doPredict", logStart=False)
    def _doPredict(self, data):
        tilePath, tileNames, tiles, outputPath = data
        if tiles:
            imgs = tiles
        else:
            imgs = []
            for tile in tileNames:
                imageFile = os.path.join(tilePath, tile)
                imgs.append(cv2.imread(imageFile))
            imgs = PILImage.create(os.path.join(tilePath, tileName))

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
