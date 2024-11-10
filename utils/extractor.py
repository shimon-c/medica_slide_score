'''

https://openslide.org/api/python/#installing
pip install openslide-python

'''

# pip install openslide-python
import argparse
from pathlib import Path
import os
import numpy
#import sys
import glob
import logging
import shutil
import threading

from multiprocessing import JoinableQueue, Process, Queue
from openslide import open_slide, OpenSlide, ImageSlide
#from PIL import Image , ImageCms
from openslide.deepzoom import DeepZoomGenerator
#from slidems.common.logi import CreateLogger, Duration


log = logging.getLogger(__name__)

class TileExtractorWorker(Process):

    def __init__(self, workerId, workQueue, resultsQueue, slide, outputPath,
                 imgFormat, stdFilter, tileSize, level, evaluateQueue=None,
                 saveTiles=True):
        Process.__init__(self, name='TileExtractorWorker')
        self.daemon = True
        self._workerId = workerId
        self._queue = workQueue
        self._ndpiSlide = slide
        self._outputPath = outputPath
        self._imgFormat = imgFormat
        self._stdFilter = stdFilter
        self._results = resultsQueue
        self.tileSize = tileSize
        self._level = level
        self._evaluateQueue = evaluateQueue
        self._tileCount = 0
        self._saveTiles = saveTiles

    #@Duration(log, msg=f"TileExtractorWorker", logStart=False)
    def run(self):
        log.debug(f"TileExtractorWorker {self._workerId} activated")
        if not os.path.isfile(self._ndpiSlide):
            log.error(f"Worker - {self._workerId} Failde to open - {self._ndpiSlide}")
            raise Exception(f"Worker - {self._workerId} Failde to open - {self._ndpiSlide}")
        _slide = open_slide(self._ndpiSlide)
        dz = DeepZoomGenerator(_slide, self.tileSize, overlap=1)
        while True:
            try:
                data = self._queue.get()
                if data is None:
                    self._queue.task_done()
                    break
                row, col = data
                self._save_tile(row, col, dz)
            except Exception as e:
                log.error(f"Exception - {e}")
            self._queue.task_done()

    def _save_tile(self, row, col, dz):
        self._tileCount += 1
        tilename = f"{col}_{row}.{self._imgFormat}"
        tilepath = os.path.join(f"{self._outputPath}/{tilename}")
        pilImage = dz.get_tile(self._level, (col, row))
        imgarr = numpy.array(pilImage, dtype=float)
        std = numpy.std(imgarr)
        if std >= self._stdFilter:
            self._results.put((f"{tilename} Ok {std} ", tilename, row,col))
            if self._evaluateQueue:
                self._evaluateQueue.put((self._outputPath, tilename, pilImage, self._outputPath))
            if self._saveTiles:
                if not os.path.exists(tilepath):
                    #log.debug(f"Saving tile {tilepath}")
                    pilImage.save(tilepath, quality=90, icc_profile=pilImage.info.get('icc_profile'))
        else:
            self._results.put(f"{tilename} Filterd {std}")



class TileExtractor(object):
    '''
    classdocs
    '''

    level = 17
    tile_size = 840

    @classmethod
    def GetTilesDir(cls, slide, outputPath):
        sourceSlideName = Path(slide).stem
        tilesDir = os.path.join(outputPath, f"{sourceSlideName}_level_{cls.level}_size_{cls.tile_size}")
        return sourceSlideName, tilesDir

    @classmethod
    def GetTileCoordinates(cls, slide, col, row):
        _slide = open_slide(slide)
        dz = DeepZoomGenerator(_slide, cls.tile_size, overlap=1)
        return dz.get_tile_coordinates(cls.level, (col,row))

    @classmethod
    def GetTileCoordinatesAtLevel(cls, slide,
                                  original_x, original_y,
                                  to_level):
        """Converts deep zoom coordinates from default to target level.
        Args:
            slide: An OpenSlide object representing the DZI.
            original_x: X coordinate at the original level (from_level).
            original_y: Y coordinate at the original level (from_level).
            from_level: Level of the original coordinates.
            to_level: Target level for the new coordinates.

        Returns:
            A tuple containing the new coordinates (x', y') at the target level.
        """
        _slide = open_slide(slide)
        highest_level = _slide.level_downsamples()
        downsample_factor = _slide.level_downsamples()[cls.level] / _slide.level_downsamples()[to_level]

        new_x = int(original_x // downsample_factor)
        new_y = int(original_y // downsample_factor)
        return new_x, new_y



    def __init__(self, slide, outputPath, img_format='jpg',
                 std_filter=10, workers=2, evaluateQueue=None,
                 saveTiles=True):
        '''
        Constructor
        '''
        self.ndpi_slide = slide
        self.outputPath = outputPath
        self.img_format = img_format
        self.std_filter = std_filter
        self.save_tiles = saveTiles
        assert os.path.isfile(self.ndpi_slide), f"NDPI file was not found at {self.ndpi_slide}"
        self.source_slide_name, self._tilesDir = self.GetTilesDir(self.ndpi_slide, self.outputPath)
        shutil.rmtree(self._tilesDir, ignore_errors=True)
        if not os.path.exists(self._tilesDir):
            os.makedirs(self._tilesDir,exist_ok=True)
            # if self.save_tiles:
            #     os.makedirs(f"{self._tilesDir}_GoodFocus")
            #     os.makedirs(f"{self._tilesDir}_BadFocus")
            #     os.makedirs(f"{self._tilesDir}_NotRelevant")
        else:
            pattern = os.path.join(self._tilesDir, f"*.{img_format}")
            tiles = glob.glob(pattern)
            existsTiles = len(tiles)
            log.debug (f"Warning will not extract, {existsTiles} tiles already exists")

        self._slide = open_slide(self.ndpi_slide)
        if not self._slide:
            raise Exception(f"Error opening slide file - {self.ndpi_slide}")
        self.dz = DeepZoomGenerator(self._slide, self.tile_size, overlap=1)
        log.debug (self.dz)
        n_levels = len(self.dz.level_tiles)
        self.totalTiles = 0
        self.cols, self.rows = 0,0
        if n_levels <= self.level:
            logging.error(f'Bad slide: {self.ndpi_slide}')
            return
        self.cols, self.rows = self.dz.level_tiles[self.level]
        self.totalTiles = self.cols * self.rows

        self._processed = 0
        self._progress = 0
        self._workers = workers
        self._workQueue = JoinableQueue(2 * self._workers)
        self._resultQueue = Queue(self.totalTiles+2)
        self._evaluateQueue = evaluateQueue
        for i in range(self._workers):
            TileExtractorWorker(workerId=i+1,
                                workQueue=self._workQueue,
                                resultsQueue=self._resultQueue,
                                slide=self.ndpi_slide,
                                outputPath=self._tilesDir,
                                imgFormat=self.img_format,
                                stdFilter=self.std_filter,
                                tileSize=self.tile_size,
                                level=self.level,
                                evaluateQueue=self._evaluateQueue,
                                saveTiles=self.save_tiles).start()
        self._extractThread = None


    @property
    def tiles_dir(self):
        return self._tilesDir

    @property
    def progress(self):
        return self._progress

    def run_asynchronic(self):
        if self._extractThread is None:
            self.extractThread = threading.Thread(target=self.run)
            log.debug(f"Start asynchronic extract tiles for slide - {self.ndpi_slide}")
            self.extractThread.start()
        else:
            log.error(f"Already running asynchronic extract tiles for slide - {self.ndpi_slide}")
            return False

    def run(self):
        log.debug(f"OpenSlide vendor: {OpenSlide.detect_format(self.ndpi_slide)}")
        if not self._slide:
            log.debug(f"Error opening slide file - {self.ndpi_slide}")
            return False

        log.debug(f"Slide Dimensions: {self._slide.dimensions} pixels")
        log.debug(f"Slide Levels: {self._slide.level_count}")
        log.debug(f"Slide Levels dimensions: {self._slide.level_dimensions}")
        log.debug(f"Slide Levels down samples: {self._slide.level_downsamples}")
        #log.debug(f"Slide Properties: {self._slide.properties}")
        log.debug(f"Total level count - {self.dz.level_count}")
        log.debug(f"Total Tiles count - {self.totalTiles}")
        log.debug(f"level tiles - {self.dz.level_tiles}")
        log.debug(f"level dimensions - {self.dz.level_dimensions}")
        log.debug(f"Extracted tile dimension at level {self.level} = {self.dz.get_tile(self.level, (0, 0)).size}")
        log.debug(f"Tiles output directory - {self._tilesDir}")

        log.info(f"Start pushing to tile extraction workers")
        for row in range(self.rows):
            for col in range(self.cols):
                self._workQueue.put((row, col))
                self._tileDone()
        self.shutdown()
        return self._writeMetaFile()

    def _tileDone(self):
        self._processed += 1
        if self._processed % 100 == 0 or self._processed == self.totalTiles:
#            columns = shutil.get_terminal_size().columns
            self._progress = round(100*self._processed/self.totalTiles)
            print(f"Progress: {self._progress}% ({self._processed}/{self.totalTiles} tiles)")

    def shutdown(self):
        for _i in range(self._workers):
            self._workQueue.put(None)
        self._workQueue.join()
        if self._extractThread:
            self._extractThread.join()
            self._extractThread = None

    def _writeMetaFile(self):
        tiles = 0
        with open(f"{self._tilesDir}_meta_data.txt", "w") as mFile:
            mFile.write(f"Source Slide: {self.ndpi_slide}\n")
            mFile.write(f"Dimensions: {self._slide.dimensions}\n")
            mFile.write(f"Levels: {self._slide.level_count}\n")
            mFile.write(f"Level dimensions: {self._slide.level_dimensions}\n")
            mFile.write(f"Level downsamples: {self._slide.level_downsamples}\n")
            mFile.write(f"Properties: {self._slide.properties}\n")
            mFile.write(f"Associated images: {self._slide.associated_images}\n")
            mFile.write(f"level count - {self.dz.level_count}\n")
            mFile.write(f"level_tiles - {self.dz.level_tiles}\n")
            mFile.write(f"level dimensions - {self.dz.level_dimensions}\n")
            mFile.write(f"tiles at level {self.dz.level_count-8} = {self.dz.get_tile(self.dz.level_count-8, (0, 0)).size}\n")
            mFile.write(f"Used Level - {self.level}\n")
            mFile.write(f"Used Level Cols Rows- {self.cols} {self.rows}\n")
            mFile.write(f"Used Level Tiles - {self.totalTiles}\n")
            mFile.write(f"output directory - {self._tilesDir}\n")
            mFile.write("Tiles statistics:\n")
            while not self._resultQueue.empty():
                res = self._resultQueue.get()
                mFile.write(f"{res}\n")
                tiles += 1
        if self.totalTiles != tiles:
            log.debug (f"Error expected - {self.totalTiles} but extracted - {tiles}")
        return tiles




def extractor_args():
    parser = argparse.ArgumentParser(description='NDPI Converter')
    parser.add_argument('-i','--input_slide', type=str,
                        default=os.path.join(Path(__file__).parent.parent.parent, "input_slide",  "2024-01-15_21.26.19.ndpi"),
                        help="Full path to the ndpi slice file. [default: %(default)s]")
    parser.add_argument('-o','--output_path', type=str,
                        default=os.path.join(Path(__file__).parent.parent.parent, "output", "tiles"),
                        help="Output image path. [default: %(default)s]")
    parser.add_argument('--format', metavar='{jpeg|png}',
                        default="jpg",
                        help="Output image format. [default: %(default)s]")
    parser.add_argument('--std', type=int,
                        default="10",
                        help="Filter images std bigger then. [default: %(default)s]")

    args, unknown = parser.parse_known_args()
    return args


def run_extractor_in_process(args=extractor_args()):
    def runExt(args):
        extractor = TileExtractor(slide=args.input_slide,
                                  outputPath=args.output_path,
                                  img_format=args.format,
                                  workers=2,
                                  std_filter=args.std)

        extractor.run()

    proc = Process(target=runExt, args=(args,))
    rc = proc.start()
    rc = proc.join()
    log.debug(f"Extracting slide complited rc={rc}")


if __name__ == '__main__':
    args = extractor_args()
    CreateLogger(name=__name__, logFile=os.path.join(args.output_path, "extractor.log"))
    workers = 4
    extractor = TileExtractor(slide=args.input_slide,
                              outputPath=args.output_path,
                              img_format=args.format,
                              workers=workers,
                              std_filter=args.std)

    extractor.run()
    print ("\nDone")

