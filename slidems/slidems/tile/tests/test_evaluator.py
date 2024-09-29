'''
Created on 2024/06/18 13:21

@Author - Dudi Levi
'''

import pytest
import glob
import shutil
import os
import time
from slidems.tile.evaluator import TilesClassifierWorker


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