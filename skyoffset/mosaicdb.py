import os
import pymongo

from andpipe import footprintdb
from difftools import CoupledPlanes  # Planar couplings
import offsettools
import blockmosaic  # common mosaic construction functions


class MosaicDB(object):
    """Database interface for a mosaic: a set of blocks."""
    def __init__(self, dbname="m31", cname="mosaics", url="localhost",
            port=27017):
        super(MosaicDB, self).__init__()
        self.dbname = dbname
        self.cname = cname
        self.url = url
        self.port = port
        
        connection = pymongo.Connection(self.url, self.port)
        self.db = connection[self.dbname]
        self.collection = self.db[self.cname]
    
    def find_mosaics(self, sel):
        """Find mosaics given a MongoDB selector."""
        docs = {}
        recs = self.collection.find(sel)
        for rec in recs:
            mosaicName = rec['_id']
            docs[mosaicName] = rec
        return recs

    def get_mosaic_doc(self, mosaicName):
        """Return document for a single, named, mosaic."""
        mosaicDoc = self.collection.find_one({"_id": mosaicName})
        return mosaicDoc

    def insert(self, mosaicDoc):
        """docstring for insert"""
        self.collection.save(mosaicDoc)

    def find_one(self, sel):
        """docstring for find_one"""
        if type(sel) is not dict:
            sel = {"_id": sel}
        return self.collection.find_one(sel)

    def update(self, sel, doc):
        """docstring for update"""
        self.collection.update(sel, doc)
