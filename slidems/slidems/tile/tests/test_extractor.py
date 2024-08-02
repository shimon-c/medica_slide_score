'''
Created on 2024/04/14 22:01

@Author - Dudi Levi
'''


#from pathlib import Path
#from typing import Literal
import pytest
import glob
import shutil
import os
import time
from slidems.tile.extractor import TileExtractor, extractor_args, run_extractor_in_process




@pytest.mark.parametrize("slide, output, expectedSlideName, expectedSlideDir", [
    ("/home/dudi/dev/patology/raw_data/good/ANON0LPGUI17B_3_1.ndpi",
     "/home/dudi/dev/patology/biofocus/output",
     "ANON0LPGUI17B_3_1",
     "/home/dudi/dev/patology/biofocus/output/ANON0LPGUI17B_3_1_level_17_size_840"),
     ])
def test_getSlidsDir(slide,
                     output,
                     expectedSlideName,
                     expectedSlideDir):
    slideName, tilesDir   = TileExtractor.GetTilesDir(slide=slide,
                                                      outputPath=output)

    assert slideName == expectedSlideName
    assert tilesDir == expectedSlideDir


@pytest.mark.parametrize("slide, output, expectedTiles, expectedSlideDir", [
    ("/home/dudi/dev/patology/raw_data/good/ANON0LPGUI17B_3_1.ndpi",
     "/home/dudi/dev/patology/biofocus/output",
     "64",
     "/home/dudi/dev/patology/biofocus/output/ANON0LPGUI17B_3_1_level_17_size_840"),
     ])
def test_extractorINProcess(slide,
                            output,
                            expectedTiles,
                            expectedSlideDir):

    slideName, tilesDir   = TileExtractor.GetTilesDir(slide=slide,
                                                      outputPath=output)
    assert tilesDir == expectedSlideDir

    extractorArgs = extractor_args()
    extractorArgs.input_slide = slide
    extractorArgs.output_path = output
    run_extractor_in_process(args=extractorArgs)

    tiles = glob.glob(f"{tilesDir}/*.*")
    assert len(tiles) == expectedTiles


@pytest.fixture
def setup(expectedSlideDir):
    print (f"Test fixture Setup")
    if os.path.isdir(expectedSlideDir):
        print (f"rm {expectedSlideDir}")
        shutil.rmtree(expectedSlideDir)
        shutil.rmtree(f"{expectedSlideDir}_BadFocus")
        shutil.rmtree(f"{expectedSlideDir}_GoodFocus")
        shutil.rmtree(f"{expectedSlideDir}_NotRelevant")

@pytest.mark.parametrize("slide, output, expectedTiles, expectedSlideDir", [
    ("/home/dudi/dev/patology/raw_data/good/ANON0LPGUI17B_3_1.ndpi",
     "/home/dudi/dev/patology/biofocus/output",
     64,
     "/home/dudi/dev/patology/biofocus/output/ANON0LPGUI17B_3_1_level_17_size_840"),
     ])
def test_extractor(slide,
                   output,
                   expectedTiles,
                   expectedSlideDir,
                   setup):

    extractor = TileExtractor(slide=slide,
                              outputPath=output)
    slideName, tilesDir = extractor.GetTilesDir(slide=slide,
                                                outputPath=output)
    assert tilesDir == expectedSlideDir

    extractor.run_asynchronic()
    while extractor.progress != 100:
        print (f"Extraction Progress = {extractor.progress}")
        time.sleep(1)


    tiles = glob.glob(f"{tilesDir}/*.*")
    assert len(tiles) == expectedTiles
