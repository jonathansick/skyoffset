#!/usr/bin/env python
# encoding: utf-8
"""
Makes difference images and represents the couplings in a MongoDB
collection.

This is essentially a spin-off from functionality in BlockDB

History
-------
2012-03-08 - Created by Jonathan Sick

"""

import pymongo
import os
import shutil
import multiprocessing

import pyfits
import owl.astromatic

from difftools import Couplings
import offsettools

def main():
    pass

def CouplingDB(dbname="skyoffset", cname="couplings", url="localhost", port=27017):
    """docstring for CouplingDB"""
    pass

class CouplingDB(object):
    """Database interface for image couplings."""
    def __init__(self, dbname="skyoffset", cname="couplings", url="localhost",
            port=27017):
        super(CouplingDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port
        
        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]

    def make_coupling(self, blockDB, sel):
        """Initialize a couplings document by computing image differences.
        The selected blocks in the blockDB will be differenced.
        """
        pass

if __name__ == '__main__':
    main()


