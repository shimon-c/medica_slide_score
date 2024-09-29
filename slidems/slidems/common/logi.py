'''
Created on Mar 15, 2024

@author: Dudi Levi
'''

import os
import sys
import time
from tempfile import NamedTemporaryFile
import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib

log = logging.getLogger(__name__)


class StdOutFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno in (logging.DEBUG, logging.INFO)


def CreateLogger(name=False, level=logging.DEBUG, logFile=False):
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger()

    for h in list(logger.handlers):
        logger.debug(f"logger handler - {h}")
        #if isinstance(h, TimedRotatingFileHandler):
        #    logger.removeHandler(h)

    logger.setLevel(level)
    #formatter = logging.Formatter('%(levelname)-8s - %(message)s  [%(processName)s %(pathname)s:%(lineno)d]')
    formatter = logging.Formatter('%(levelname)-8s - %(message)s  [%(processName)s %(filename)s:%(lineno)d]')
    strmHanlr = logging.StreamHandler(sys.stdout)
    strmHanlr.setLevel(level)
    strmHanlr.addFilter(StdOutFilter())
    strmHanlr.setFormatter(formatter)
    logger.addHandler(strmHanlr)
    # strmHanlr = logging.StreamHandler(sys.stderr)
    # strmHanlr.setLevel(logging.WARNING)
    # strmHanlr.setFormatter(formatter)
    # logger.addHandler(strmHanlr)

    logger.propagate = False

    if not logFile:
        logFile = NamedTemporaryFile(suffix=f"_{name}.log", dir=None, delete=True, mode='w+b').name

    pathlib.Path(os.path.dirname(logFile)).mkdir(parents=True, exist_ok=True)
    fh = TimedRotatingFileHandler(logFile, when="midnight")
    fh.setLevel(level)
    formatter = logging.Formatter('%(levelname)-8s %(asctime)s : %(message)s  [%(processName)s %(filename)s:%(lineno)d]')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Logger Created - {logFile}")
    return logger



def Duration(log, msg="", logStart=True):
    def duration_child(func):
        def duration_wrapper(self,*args, **kwargs):
            startTime = time.perf_counter()
            try:
                if logStart: log.debug(f"START: {msg}")
                result = func(self,*args, **kwargs)
                totalTime = time.perf_counter() - startTime
            except Exception as e:
                totalTime = time.perf_counter() - startTime
                log.error(f"END: {msg} (Time: {totalTime:.3f} Sec)")
                raise e
            log.debug(f"END: {msg} (Time: {totalTime:.3f} Sec)")
            return result
        return duration_wrapper
    return duration_child
