'''
Created on 2024/04/14 21:15

@Author - Dudi Levi
'''

import logging
import os
from pathlib import Path
import orjson

log = logging.getLogger(__name__)



class Metadata():
    def __init__(self, file=None) -> None:
        self.file = file
        if file:
            self.read()
        else:
            self._data = {  "source slide info": {"Slide": "",
                                                  "Tiles input dir": ""},
                            "Evaluation": {"Evaluate results": "",
                                           "Slide classified as": ""}
                         } 

    @property
    def data(self):
        return self._data

    def read(self, file=None):
        if file:
            path = Path(file)
        else:
            path = Path(self.file)

        self._data = orjson.loads(path.read_bytes())

    def save(self, file=None, append=False):
        fileName = os.path.basename(file)
        filePath = os.path.dirname(file)

        mode = 'wb'
        if append:
            mode = 'ab'
        try:
            if not os.path.exists(filePath):
                log.debug(f"Creating write directory - {filePath}")
                os.makedirs(filePath)
            
            with open(file, mode) as jsonFile:
                jsonFile.write(orjson.dumps(self._data, 
                                            option=orjson.OPT_INDENT_2))

        except OSError:
            log.critical(f"Failed to write {file}")
            raise
    
    @property
    def slideName(self):
        return self._data["source slide info"]["Slide"]
    @slideName.setter
    def slideName(self, val):
        self._data["source slide info"]["Slide"] = val
    
    @property
    def tilesInputDir(self):
        return self._data["source slide info"]["Tiles input dir"]
    @tilesInputDir.setter
    def tilesInputDir(self, val):
        self._data["source slide info"]["Tiles input dir"] = val   

    @property
    def evaluationResults(self):
        return self._data["Evaluation"]["Evaluate results"]
    @evaluationResults.setter
    def evaluationResults(self, val):
        self._data["Evaluation"]["Evaluate results"] = val

    @property
    def slideClassification(self):
        return self._data["Evaluation"]["Slide classified as"]
    @slideClassification.setter
    def slideClassification(self, val):
        self._data["Evaluation"]["Slide classified as"] = val
    
    