'''
Created on 2024/04/14 22:01

@Author - Dudi Levi
'''


import os
from pathlib import Path
import pytest
from slidems.common.metadata_io import Metadata


class TestMetadata:
    testJson = os.path.join(os.path.dirname(__file__), "metadata_test.json")
    wirteTestFile = os.path.join(Path(__file__).parent.parent.parent.parent,
                                 "output", "write_test.json")

    def test_init(self):
        """
        init  file
        """
        meta = Metadata()
        assert meta

    def test_read(self):
        """
        read json file
        """
        meta = Metadata(file=self.testJson)
        meta2 = Metadata()
        meta2.read(file=self.testJson)
        assert meta.data == meta2.data
        print (meta2.data)

    def test_write(self):
        meta = Metadata(file=self.testJson)
        print(meta.data)
        meta.save(self.wirteTestFile)
        assert os.path.isfile(self.wirteTestFile) == True

    def test_set_slideName(self):
        meta = Metadata(file=self.testJson)
        org = meta.slideName
        print(f"slideName = {meta.slideName}")
        meta.slideName = "/dudi/dudi/kk.ndpi"
        assert org != meta.slideName
        assert meta.slideName == "/dudi/dudi/kk.ndpi"
        meta.save(self.wirteTestFile)