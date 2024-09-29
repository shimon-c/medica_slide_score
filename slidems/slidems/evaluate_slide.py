'''
slidems -- Evaluate Biopsy slide focus quality

@author:     Dudi Levi

@copyright:  2024 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import os
import glob
import time
import logging
from pathlib import Path
from multiprocessing import JoinableQueue
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from slidems.tile.extractor import TileExtractor
from slidems.slide.evaluate_focus import CheckTilesFocus
from slidems.common.logi import CreateLogger

log = logging.getLogger(__name__)

program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
program_license = '''%s

  Created by Dudi Levi .
  Copyright 2024 Yofi Tools. All rights reserved.

USAGE
''' % (program_shortdesc)


def EvaluateSlide_args():
    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-i','--slide_file', type=str,
                        default=os.path.join(Path(__file__).parent.parent, "input_slide",  "2024-01-15_21.26.19.ndpi"),
                        help="Full path to the ndpi slice file. [default: %(default)s]")
    parser.add_argument('-o','--output_path', type=str,
                        default=os.path.join(Path(__file__).parent.parent, "output"),
                        help="Output image path. [default: %(default)s]")
    parser.add_argument("--save_tiles", default=False, action="store_true",
                        help="Save all tiles sorted them. [default: %(default)s]")

    # Process arguments
    args, unknown = parser.parse_known_args()
    return args


def EvaluateSlide(args=None):
    programName = os.path.basename(__file__).split('.')[0]
    CreateLogger(logFile=os.path.join(args.output_path, f"{programName}.log"))
    log.debug(sys.argv)

    log.info(f"Starting: {programName}")

    try:
        # Setup argument parser
        if os.path.isdir(args.slide_file):
            slide_files = glob.glob(os.path.join(args.slide_file, "*.ndpi"))
        else:
            slide_files = [args.slide_file]

        workers = 1
        evaluateQueue = JoinableQueue(2 * workers)
        for slide in slide_files:
            extractore = TileExtractor(slide=slide,
                                       outputPath=args.output_path,
                                       workers=workers,
                                       evaluateQueue=evaluateQueue,
                                       saveTiles=args.save_tiles)
            evaluate = CheckTilesFocus(slide=slide,
                                       outputPath=args.output_path,
                                       checkTilesQueud=evaluateQueue,
                                       workers=workers,
                                       saveResults=args.save_tiles)

            evaluate.run()
            extractore.run()
            evaluate.shutdown(extractore.tiles_dir)


        # for inpath in args.path:
        #     ### do something with inpath ###
        #     print(inpath)
        return 0
    except Exception as e:
        print(e)
        sys.stderr.write(programName + ": " + repr(e) + "\n")
        raise(e)
        return 2


if __name__ == '__main__':
    print(sys.argv)
    args = EvaluateSlide_args()
    sys.exit(EvaluateSlide(args))

